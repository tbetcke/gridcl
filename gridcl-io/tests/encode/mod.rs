//! A binary MSH 4.1 encoder of the file model, written for the tests from the format
//! description, independently of gridcl-io's writer: it makes byte-swapped and 4-byte-`size_t`
//! copies of the corpus (docs/plans/M4.md, T6).

use gridcl_io::gmsh::{Entity, MshFile, PartitionedEntity};

/// Binary values in one byte order and `size_t` width.
struct Out {
    bytes: Vec<u8>,
    big_endian: bool,
    size_t: usize,
}

impl Out {
    fn text(&mut self, s: &str) {
        self.bytes.extend_from_slice(s.as_bytes());
    }
    fn int(&mut self, x: i32) {
        let b = if self.big_endian {
            x.to_be_bytes()
        } else {
            x.to_le_bytes()
        };
        self.bytes.extend_from_slice(&b);
    }
    fn size(&mut self, x: u64) {
        if self.size_t == 4 {
            let x = u32::try_from(x).expect("fits a 4-byte size_t");
            let b = if self.big_endian {
                x.to_be_bytes()
            } else {
                x.to_le_bytes()
            };
            self.bytes.extend_from_slice(&b);
        } else {
            let b = if self.big_endian {
                x.to_be_bytes()
            } else {
                x.to_le_bytes()
            };
            self.bytes.extend_from_slice(&b);
        }
    }
    fn len(&mut self, n: usize) {
        self.size(n as u64);
    }
    fn float(&mut self, x: f64) {
        let b = if self.big_endian {
            x.to_be_bytes()
        } else {
            x.to_le_bytes()
        };
        self.bytes.extend_from_slice(&b);
    }
    fn ints(&mut self, xs: &[i32]) {
        self.len(xs.len());
        for &x in xs {
            self.int(x);
        }
    }
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
    }
}

/// `file` as a binary MSH 4.1 file in the given byte order and `size_t` width.
pub fn encode_binary(file: &MshFile, big_endian: bool, size_t: usize) -> Vec<u8> {
    let mut o = Out {
        bytes: Vec::new(),
        big_endian,
        size_t,
    };
    o.text(&format!("$MeshFormat\n4.1 1 {size_t}\n"));
    o.int(1);
    o.text("\n$EndMeshFormat\n");
    if !file.physical_names.is_empty() {
        o.text(&format!("$PhysicalNames\n{}\n", file.physical_names.len()));
        for n in &file.physical_names {
            o.text(&format!("{} {} \"{}\"\n", n.dim, n.tag, n.name));
        }
        o.text("$EndPhysicalNames\n");
    }
    if let Some(e) = &file.entities {
        o.text("$Entities\n");
        let lists = [&e.points, &e.curves, &e.surfaces, &e.volumes];
        for l in lists {
            o.len(l.len());
        }
        for (dim, l) in lists.into_iter().enumerate() {
            for x in l {
                o.int(x.tag);
                o.entity(x, dim);
            }
        }
        o.text("\n$EndEntities\n");
    }
    if let Some(p) = &file.partitioned_entities {
        o.text("$PartitionedEntities\n");
        o.len(p.num_partitions);
        o.len(p.ghost_entities.len());
        for &(t, q) in &p.ghost_entities {
            o.int(t);
            o.int(q);
        }
        let lists: [&Vec<PartitionedEntity>; 4] = [&p.points, &p.curves, &p.surfaces, &p.volumes];
        for l in lists {
            o.len(l.len());
        }
        for (dim, l) in lists.into_iter().enumerate() {
            for x in l {
                o.int(x.entity.tag);
                o.int(x.parent_dim as i32);
                o.int(x.parent_tag);
                o.ints(&x.partitions);
                o.entity(&x.entity, dim);
            }
        }
        o.text("\n$EndPartitionedEntities\n");
    }
    let node_tags = file.node_blocks.iter().flat_map(|b| b.tags.iter().copied());
    let num_nodes = node_tags.clone().count();
    o.text("$Nodes\n");
    o.len(file.node_blocks.len());
    o.len(num_nodes);
    o.size(node_tags.clone().min().unwrap_or(0) as u64);
    o.size(node_tags.max().unwrap_or(0) as u64);
    for b in &file.node_blocks {
        o.int(b.dim as i32);
        o.int(b.entity_tag);
        o.int(0);
        o.len(b.tags.len());
        for &t in &b.tags {
            o.size(t as u64);
        }
        for x in b.coords.iter().flatten() {
            o.float(*x);
        }
    }
    o.text("\n$EndNodes\n");
    let element_tags = file
        .element_blocks
        .iter()
        .flat_map(|b| b.tags.iter().copied());
    o.text("$Elements\n");
    o.len(file.element_blocks.len());
    o.len(element_tags.clone().count());
    o.size(element_tags.clone().min().unwrap_or(0) as u64);
    o.size(element_tags.max().unwrap_or(0) as u64);
    for b in &file.element_blocks {
        o.int(b.dim as i32);
        o.int(b.entity_tag);
        o.int(b.element_type);
        o.len(b.tags.len());
        let k = b.nodes.len() / b.tags.len().max(1);
        for (i, &t) in b.tags.iter().enumerate() {
            o.size(t as u64);
            for &n in &b.nodes[i * k..(i + 1) * k] {
                o.size(n as u64);
            }
        }
    }
    o.text("\n$EndElements\n");
    if !file.periodic.is_empty() {
        o.text("$Periodic\n");
        o.len(file.periodic.len());
        for l in &file.periodic {
            o.int(l.dim as i32);
            o.int(l.tag);
            o.int(l.master_tag);
            o.len(l.affine.len());
            for &a in &l.affine {
                o.float(a);
            }
            o.len(l.nodes.len());
            for &(a, b) in &l.nodes {
                o.size(a as u64);
                o.size(b as u64);
            }
        }
        o.text("\n$EndPeriodic\n");
    }
    o.bytes
}
