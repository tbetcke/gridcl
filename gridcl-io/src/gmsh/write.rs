//! The MSH 4.1 writer: ASCII with shortest round-trip floats, or binary (little-endian, 8-byte
//! `size_t`) in the layout Gmsh writes.

use std::io::Write;

use gridcl_device::{GridError, Result};

use super::{Entity, FORMAT, MshFile, PartitionedEntity};

/// The encoding of a written file. Binary files are little-endian with an 8-byte `size_t`.
///
/// # Example
///
/// ```
/// use gridcl_io::gmsh::{Encoding, MshFile};
///
/// for (encoding, header) in [(Encoding::Ascii, "4.1 0 8"), (Encoding::Binary, "4.1 1 8")] {
///     let mut bytes = Vec::new();
///     MshFile::default().write_to(&mut bytes, encoding)?;
///     assert!(bytes.starts_with(format!("$MeshFormat\n{header}\n").as_bytes()));
/// }
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// Text, with shortest round-trip floats.
    Ascii,
    /// Binary, little-endian, 8-byte `size_t`.
    Binary,
}

/// The bytes of a file being written: values as text tokens separated by spaces, one record
/// per line, or as little-endian binary values; section headers and `$PhysicalNames` are
/// text in both.
struct Out {
    bytes: Vec<u8>,
    binary: bool,
    /// Whether the next text token starts a line.
    line_start: bool,
}

impl Out {
    fn text(&mut self, s: &str) {
        self.bytes.extend_from_slice(s.as_bytes());
    }

    /// One value: a token in ASCII, `binary` in binary files.
    fn value(&mut self, token: std::fmt::Arguments<'_>, binary: &[u8]) {
        if self.binary {
            self.bytes.extend_from_slice(binary);
        } else {
            if !self.line_start {
                self.bytes.push(b' ');
            }
            self.bytes.write_fmt(token).expect("writing to a Vec");
            self.line_start = false;
        }
    }

    fn int(&mut self, x: i32) {
        self.value(format_args!("{x}"), &x.to_le_bytes());
    }

    fn size(&mut self, x: u64) {
        self.value(format_args!("{x}"), &x.to_le_bytes());
    }

    fn len(&mut self, n: usize) {
        self.size(n as u64);
    }

    /// A node or element tag, ≥ 1 by `check_writable`.
    fn tag(&mut self, t: i64) {
        self.size(t as u64);
    }

    fn dim(&mut self, d: usize) {
        self.int(i32::try_from(d).expect("a dimension 0-3"));
    }

    /// A `double`, in ASCII as the shortest decimal that reads back to the same value.
    fn float(&mut self, x: f64) {
        let a = x.abs();
        if x == 0.0 || (1e-5..1e16).contains(&a) || !x.is_finite() {
            self.value(format_args!("{x}"), &x.to_le_bytes());
        } else {
            self.value(format_args!("{x:e}"), &x.to_le_bytes());
        }
    }

    fn ints(&mut self, xs: &[i32]) {
        self.len(xs.len());
        for &x in xs {
            self.int(x);
        }
    }

    /// Ends a record: a line break in ASCII.
    fn end_line(&mut self) {
        if !self.binary {
            self.bytes.push(b'\n');
            self.line_start = true;
        }
    }

    /// The header line of a section.
    fn begin(&mut self, section: &str) {
        self.text(&format!("${section}\n"));
        self.line_start = true;
    }

    /// The end line of a section; binary data is followed by a line break first.
    fn end(&mut self, section: &str) {
        if self.binary {
            self.bytes.push(b'\n');
        }
        self.text(&format!("$End{section}\n"));
    }

    /// The coordinates or bounding box, physical tags and bounding entities of an entity of
    /// dimension `dim`.
    fn entity(&mut self, e: &Entity, dim: usize) {
        for x in e.min {
            self.float(x);
        }
        if dim > 0 {
            for x in e.max {
                self.float(x);
            }
        }
        self.ints(&e.physical_tags);
        if dim > 0 {
            self.ints(&e.boundary);
        }
        self.end_line();
    }
}

/// `Parse` for a model whose file the parser would reject; a model has no byte offset.
fn unwritable(message: String) -> GridError {
    GridError::Parse {
        format: FORMAT,
        offset: 0,
        message: format!("{message} (no byte offset: the model, not a file)"),
    }
}

impl MshFile {
    /// Fails unless the parser can read the written file: node and element tags of 1 or more
    /// (the header range of an empty section aside), names without quotes or line breaks.
    fn check_writable(&self) -> Result<()> {
        for n in &self.physical_names {
            if n.name.contains(['"', '\n', '\r']) {
                return Err(unwritable(format!(
                    "physical name {:?} has a quote or a line break",
                    n.name
                )));
            }
        }
        let node_tags = self.node_blocks.iter().flat_map(|b| b.tags.iter());
        let element_nodes = self.element_blocks.iter().flat_map(|b| b.nodes.iter());
        let periodic_nodes = self
            .periodic
            .iter()
            .flat_map(|l| l.nodes.iter().flat_map(|(a, b)| [a, b]));
        let element_tags = self.element_blocks.iter().flat_map(|b| b.tags.iter());
        let tags = node_tags
            .chain(element_nodes)
            .chain(periodic_nodes)
            .map(|t| ("node", t))
            .chain(element_tags.map(|t| ("element", t)));
        for (what, &t) in tags {
            if t < 1 {
                return Err(unwritable(format!(
                    "{what} tag {t}: tags must be 1 to 2^63 - 1"
                )));
            }
        }
        Ok(())
    }

    /// Writes the file as MSH 4.1 in `encoding`: `$MeshFormat`, then `$PhysicalNames`,
    /// `$Entities` and `$PartitionedEntities` when present, `$Nodes`, `$Elements`, and
    /// `$Periodic` when present. The header counts and tag ranges of `$Nodes` and `$Elements`
    /// are computed from the blocks; node blocks are written without parametric coordinates.
    ///
    /// # Errors
    ///
    /// `Parse` (no byte offset) before anything is written, for a model the parser would
    /// reject: a node or element tag below 1, or a physical name with a quote or a line break;
    /// `Io` if writing fails.
    ///
    /// # Panics
    ///
    /// If an element block's node list is not a whole number of elements.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::gmsh::{ElementBlock, Encoding, MshFile, NodeBlock, parse};
    ///
    /// let file = MshFile {
    ///     node_blocks: vec![NodeBlock {
    ///         dim: 2,
    ///         entity_tag: 1,
    ///         tags: vec![1, 2, 3],
    ///         coords: vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 1e-30, 0.0]],
    ///     }],
    ///     element_blocks: vec![ElementBlock {
    ///         dim: 2,
    ///         entity_tag: 1,
    ///         element_type: 2,
    ///         tags: vec![7],
    ///         nodes: vec![1, 2, 3],
    ///     }],
    ///     ..MshFile::default()
    /// };
    /// for encoding in [Encoding::Ascii, Encoding::Binary] {
    ///     let mut bytes = Vec::new();
    ///     file.write_to(&mut bytes, encoding)?;
    ///     assert_eq!(parse(&bytes)?, file);
    /// }
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn write_to(&self, w: &mut dyn Write, encoding: Encoding) -> Result<()> {
        self.check_writable()?;
        let binary = encoding == Encoding::Binary;
        let mut o = Out {
            bytes: Vec::new(),
            binary,
            line_start: true,
        };
        o.begin("MeshFormat");
        o.text(if binary { "4.1 1 8\n" } else { "4.1 0 8\n" });
        if binary {
            o.bytes.extend_from_slice(&1i32.to_le_bytes());
        }
        o.end("MeshFormat");

        if !self.physical_names.is_empty() {
            o.begin("PhysicalNames");
            o.text(&format!("{}\n", self.physical_names.len()));
            for n in &self.physical_names {
                o.text(&format!("{} {} \"{}\"\n", n.dim, n.tag, n.name));
            }
            o.text("$EndPhysicalNames\n");
        }

        if let Some(e) = &self.entities {
            o.begin("Entities");
            let lists = [&e.points, &e.curves, &e.surfaces, &e.volumes];
            for l in lists {
                o.len(l.len());
            }
            o.end_line();
            for (dim, l) in lists.into_iter().enumerate() {
                for x in l {
                    o.int(x.tag);
                    o.entity(x, dim);
                }
            }
            o.end("Entities");
        }

        if let Some(p) = &self.partitioned_entities {
            o.begin("PartitionedEntities");
            o.len(p.num_partitions);
            o.end_line();
            o.len(p.ghost_entities.len());
            o.end_line();
            for &(tag, partition) in &p.ghost_entities {
                o.int(tag);
                o.int(partition);
                o.end_line();
            }
            let lists: [&Vec<PartitionedEntity>; 4] =
                [&p.points, &p.curves, &p.surfaces, &p.volumes];
            for l in lists {
                o.len(l.len());
            }
            o.end_line();
            for (dim, l) in lists.into_iter().enumerate() {
                for x in l {
                    o.int(x.entity.tag);
                    o.dim(x.parent_dim);
                    o.int(x.parent_tag);
                    o.ints(&x.partitions);
                    o.entity(&x.entity, dim);
                }
            }
            o.end("PartitionedEntities");
        }

        let node_tags = self.node_blocks.iter().flat_map(|b| b.tags.iter().copied());
        o.begin("Nodes");
        o.len(self.node_blocks.len());
        o.len(node_tags.clone().count());
        o.tag(node_tags.clone().min().unwrap_or(0));
        o.tag(node_tags.max().unwrap_or(0));
        o.end_line();
        for b in &self.node_blocks {
            o.dim(b.dim);
            o.int(b.entity_tag);
            o.int(0);
            o.len(b.tags.len());
            o.end_line();
            for &t in &b.tags {
                o.tag(t);
                o.end_line();
            }
            for x in &b.coords {
                for &c in x {
                    o.float(c);
                }
                o.end_line();
            }
        }
        o.end("Nodes");

        let element_tags = self
            .element_blocks
            .iter()
            .flat_map(|b| b.tags.iter().copied());
        o.begin("Elements");
        o.len(self.element_blocks.len());
        o.len(element_tags.clone().count());
        o.tag(element_tags.clone().min().unwrap_or(0));
        o.tag(element_tags.max().unwrap_or(0));
        o.end_line();
        for b in &self.element_blocks {
            o.dim(b.dim);
            o.int(b.entity_tag);
            o.int(b.element_type);
            o.len(b.tags.len());
            o.end_line();
            assert!(
                b.tags.is_empty() && b.nodes.is_empty()
                    || !b.tags.is_empty() && b.nodes.len().is_multiple_of(b.tags.len()),
                "write_to: {} node tags are not whole elements of {} elements",
                b.nodes.len(),
                b.tags.len()
            );
            let k = b.nodes.len() / b.tags.len().max(1);
            for (i, &t) in b.tags.iter().enumerate() {
                o.tag(t);
                for &n in &b.nodes[i * k..(i + 1) * k] {
                    o.tag(n);
                }
                o.end_line();
            }
        }
        o.end("Elements");

        if !self.periodic.is_empty() {
            o.begin("Periodic");
            o.len(self.periodic.len());
            o.end_line();
            for l in &self.periodic {
                o.dim(l.dim);
                o.int(l.tag);
                o.int(l.master_tag);
                o.end_line();
                o.len(l.affine.len());
                for &a in &l.affine {
                    o.float(a);
                }
                o.end_line();
                o.len(l.nodes.len());
                o.end_line();
                for &(a, b) in &l.nodes {
                    o.tag(a);
                    o.tag(b);
                    o.end_line();
                }
            }
            o.end("Periodic");
        }
        w.write_all(&o.bytes)
            .map_err(|e| GridError::Io(format!("writing a Gmsh file: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmsh::{
        ElementBlock, Entities, NodeBlock, PartitionedEntities, Periodic, PhysicalName, parse,
    };
    use proptest::prelude::*;

    fn round_trip(file: &MshFile) {
        for encoding in [Encoding::Ascii, Encoding::Binary] {
            let mut bytes = Vec::new();
            file.write_to(&mut bytes, encoding).unwrap();
            let back = parse(&bytes).unwrap_or_else(|e| panic!("{encoding:?}: {e}"));
            assert_eq!(&back, file, "{encoding:?}");
        }
    }

    #[test]
    fn write_floats_round_trip_shortest() {
        let xs = [
            0.0,
            -0.0,
            0.1,
            1.0 / 3.0,
            -2.5e-7,
            1e-300,
            f64::MIN_POSITIVE,
            5e-324,
            1.7976931348623157e308,
            123456789.125,
            1e16,
        ];
        for x in xs {
            let mut o = Out {
                bytes: Vec::new(),
                binary: false,
                line_start: true,
            };
            o.float(x);
            let text = String::from_utf8(o.bytes).unwrap();
            let back: f64 = text.parse().unwrap();
            assert_eq!(back.to_bits(), x.to_bits(), "{x} written as {text}");
            assert!(text.len() <= 24, "{x} written as {text}");
        }
    }

    #[test]
    fn write_every_section() {
        let entity = |tag, dim: usize| Entity {
            tag,
            min: [0.0, -1.5, 2.0],
            max: if dim == 0 {
                [0.0, -1.5, 2.0]
            } else {
                [1.0, 0.5, 2.0]
            },
            physical_tags: vec![3, 4],
            boundary: if dim == 0 { vec![] } else { vec![1, -2] },
        };
        let file = MshFile {
            physical_names: vec![PhysicalName {
                dim: 1,
                tag: 3,
                name: "two words".into(),
            }],
            entities: Some(Entities {
                points: vec![entity(1, 0)],
                curves: vec![entity(2, 1), entity(3, 1)],
                surfaces: vec![],
                volumes: vec![entity(9, 3)],
            }),
            partitioned_entities: Some(PartitionedEntities {
                num_partitions: 2,
                ghost_entities: vec![(5, 1)],
                points: vec![PartitionedEntity {
                    entity: entity(4, 0),
                    parent_dim: 0,
                    parent_tag: 1,
                    partitions: vec![1, 2],
                }],
                curves: vec![],
                surfaces: vec![PartitionedEntity {
                    entity: entity(6, 2),
                    parent_dim: 2,
                    parent_tag: 1,
                    partitions: vec![2],
                }],
                volumes: vec![],
            }),
            node_blocks: vec![
                NodeBlock {
                    dim: 0,
                    entity_tag: 1,
                    tags: vec![10],
                    coords: vec![[0.0, 0.0, 0.0]],
                },
                NodeBlock {
                    dim: 2,
                    entity_tag: 4,
                    tags: vec![],
                    coords: vec![],
                },
                NodeBlock {
                    dim: 1,
                    entity_tag: 2,
                    tags: vec![3, 1 << 40],
                    coords: vec![[0.25, 1e-20, -3.0], [1.0, 2.0, 3.0]],
                },
            ],
            element_blocks: vec![
                ElementBlock {
                    dim: 1,
                    entity_tag: 2,
                    element_type: 8,
                    tags: vec![4, 2],
                    nodes: vec![10, 3, 1 << 40, 3, 10, 1 << 40],
                },
                ElementBlock {
                    dim: 0,
                    entity_tag: 1,
                    element_type: 15,
                    tags: vec![],
                    nodes: vec![],
                },
            ],
            periodic: vec![
                Periodic {
                    dim: 1,
                    tag: 2,
                    master_tag: 3,
                    affine: (0..16).map(f64::from).collect(),
                    nodes: vec![(3, 10)],
                },
                Periodic {
                    dim: 0,
                    tag: 1,
                    master_tag: 1,
                    affine: vec![],
                    nodes: vec![],
                },
            ],
        };
        round_trip(&file);
        round_trip(&MshFile::default());
    }

    /// Binary files have Gmsh's header: version, file type 1, data size 8, the int 1.
    #[test]
    fn write_binary_header() {
        let mut bytes = Vec::new();
        MshFile::default()
            .write_to(&mut bytes, Encoding::Binary)
            .unwrap();
        assert!(bytes.starts_with(b"$MeshFormat\n4.1 1 8\n\x01\x00\x00\x00\n$EndMeshFormat\n"));
        let mut text = Vec::new();
        MshFile::default()
            .write_to(&mut text, Encoding::Ascii)
            .unwrap();
        assert!(text.starts_with(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"));
    }

    /// Models the parser would reject give `Parse` and write nothing: tags below 1 (nodes,
    /// element nodes, elements, periodic nodes) and names with a quote or a line break.
    #[test]
    fn write_rejects_unparsable_models() {
        let good = MshFile {
            physical_names: vec![PhysicalName {
                dim: 2,
                tag: 1,
                name: "domain".into(),
            }],
            node_blocks: vec![NodeBlock {
                dim: 2,
                entity_tag: 1,
                tags: vec![1, 2, 3],
                coords: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }],
            element_blocks: vec![ElementBlock {
                dim: 2,
                entity_tag: 1,
                element_type: 2,
                tags: vec![1],
                nodes: vec![1, 2, 3],
            }],
            periodic: vec![Periodic {
                dim: 1,
                tag: 2,
                master_tag: 1,
                affine: vec![],
                nodes: vec![(2, 1)],
            }],
            ..MshFile::default()
        };
        round_trip(&good);
        type Edit = (&'static str, fn(&mut MshFile));
        let edits: [Edit; 6] = [
            ("node tag", |f| f.node_blocks[0].tags[0] = 0),
            ("element node", |f| f.element_blocks[0].nodes[1] = -4),
            ("element tag", |f| f.element_blocks[0].tags[0] = i64::MIN),
            ("periodic node", |f| f.periodic[0].nodes[0].1 = 0),
            ("quoted name", |f| f.physical_names[0].name = "a\"b".into()),
            ("two-line name", |f| {
                f.physical_names[0].name = "a\nb".into()
            }),
        ];
        for (what, edit) in edits {
            let mut bad = good.clone();
            edit(&mut bad);
            for encoding in [Encoding::Ascii, Encoding::Binary] {
                let mut bytes = Vec::new();
                let r = bad.write_to(&mut bytes, encoding);
                assert!(
                    matches!(r, Err(GridError::Parse { offset: 0, .. })),
                    "{what}: {r:?}"
                );
                assert!(bytes.is_empty(), "{what}: bytes written");
            }
        }
    }

    /// A small model: one node block and one element block of type `ty` (A.7), random tags
    /// and coordinates.
    fn small_model() -> impl Strategy<Value = MshFile> {
        let types = prop::sample::select(vec![15, 1, 8, 2, 9, 3, 16, 4, 5, 6, 7, 14, 118]);
        (
            types,
            1usize..6,
            prop::collection::vec(any::<f64>().prop_filter("finite", |x| x.is_finite()), 3),
            1i64..i64::MAX / 2,
            prop::collection::vec(1i32..100, 0..3),
        )
            .prop_map(|(ty, n, xyz, base, physical)| {
                let k = crate::gmsh::element_type(ty).unwrap().num_nodes;
                let num_nodes = k + 2;
                let tags: Vec<i64> = (0..num_nodes as i64).map(|i| base + 3 * i).collect();
                let coords = (0..num_nodes)
                    .map(|i| [xyz[0] * i as f64, xyz[1], xyz[2] / (i + 1) as f64])
                    .collect();
                let nodes = (0..n * k).map(|i| tags[i % num_nodes]).collect();
                let dim = gridcl_refcell::tdim(crate::gmsh::element_type(ty).unwrap().cell);
                MshFile {
                    physical_names: physical
                        .iter()
                        .map(|&tag| PhysicalName {
                            dim,
                            tag,
                            name: format!("group {tag}"),
                        })
                        .collect(),
                    node_blocks: vec![NodeBlock {
                        dim,
                        entity_tag: 1,
                        tags,
                        coords,
                    }],
                    element_blocks: vec![ElementBlock {
                        dim,
                        entity_tag: 1,
                        element_type: ty,
                        tags: (1..=n as i64).map(|i| base + i).collect(),
                        nodes,
                    }],
                    ..MshFile::default()
                }
            })
    }

    proptest! {
        #[test]
        fn write_parse_round_trips(file in small_model()) {
            for encoding in [Encoding::Ascii, Encoding::Binary] {
                let mut bytes = Vec::new();
                file.write_to(&mut bytes, encoding).unwrap();
                prop_assert_eq!(&parse(&bytes).unwrap(), &file);
            }
        }
    }
}
