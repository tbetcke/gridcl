//! The figures of the book's element definitions, generated from tetraloma-refcell's tables: an
//! SVG of each reference cell with its vertex, edge and face numbers, and the Lagrange node
//! orderings for p = 1, 2, 3 on the interval, triangle and quadrilateral and p = 1, 2 on the
//! tetrahedron, hexahedron, prism and pyramid.
//!
//! Every coordinate, sub-entity list and node position comes from tetraloma-refcell
//! ([`vertices`], [`sub_entities`], [`lagrange_points`], [`lagrange_entity_nodes`]); nothing
//! is retyped here. Three-dimensional cells are drawn in one fixed oblique projection: the
//! point (x, y, z) is drawn at (x + 0.75 y, z + 0.45 y), so the x–z plane keeps its shape and
//! the y axis recedes up and to the right. Edges and labels on faces turned away from the
//! viewer are dashed or faded.
//!
//! The output is deterministic: pixel coordinates are printed with one decimal and the files
//! come in a fixed order. The binary `figures` writes them to `docs/user/src/figures/`; the test
//! `figures_are_generated` checks that the committed files equal [`svg_files`].

use tetraloma::CellType;
use tetraloma_refcell::{
    NodeFamily, lagrange_entity_nodes, lagrange_points, num_sub_entities, sub_entities, tdim,
    vertices,
};

/// The cells of the figures with their names in file names and titles, in the order of the
/// book.
const CELLS: [(CellType, &str); 7] = [
    (CellType::Interval, "interval"),
    (CellType::Triangle, "triangle"),
    (CellType::Quadrilateral, "quadrilateral"),
    (CellType::Tetrahedron, "tetrahedron"),
    (CellType::Hexahedron, "hexahedron"),
    (CellType::Prism, "prism"),
    (CellType::Pyramid, "pyramid"),
];

/// The horizontal shift of the oblique projection per unit of y.
const DEPTH_X: f64 = 0.75;
/// The vertical shift of the oblique projection per unit of y.
const DEPTH_Y: f64 = 0.45;
/// The direction in which the viewer looks: every point on a line in this direction is drawn
/// at the same place, and a face whose outward normal points against it faces the viewer.
const VIEW: [f64; 3] = [-DEPTH_X, 1.0, -DEPTH_Y];

/// Pixels per unit of reference length.
const SCALE: f64 = 240.0;
/// The space around the drawing, for labels.
const MARGIN: f64 = 48.0;
/// The height of the legend below the drawing.
const LEGEND_HEIGHT: f64 = 48.0;
/// The smallest width of a figure, so that the legend fits.
const MIN_WIDTH: f64 = 420.0;
/// The distance of a vertex label from its vertex, in pixels.
const VERTEX_LABEL_OFFSET: f64 = 17.0;

/// The roles of labels and nodes: their CSS class and their name in the legend.
const ROLES: [(&str, &str); 4] = [
    ("v", "vertex"),
    ("e", "edge"),
    ("f", "face"),
    ("i", "interior"),
];

/// The style sheet of every figure: a white background (the book has dark themes), labels with
/// a white halo, one colour per role, and dashed or faded hidden parts.
const STYLE: &str = "text{font-family:sans-serif;font-size:15px;text-anchor:middle;\
paint-order:stroke;stroke:#fff;stroke-width:4px;stroke-linejoin:round}\
text.n{font-size:13px;text-anchor:start}text.k{text-anchor:start;fill:#1a1a1a}\
.v{fill:#1a1a1a;font-weight:bold}.e{fill:#1764ab}.f{fill:#b5480b;font-style:italic}\
.i{fill:#6a3d9a}.h{opacity:0.45}\
line,path.a{stroke:#1a1a1a;stroke-width:2;fill:none}line.l{stroke:#8c8c8c;stroke-width:1.5}\
line.h{stroke-dasharray:6 4;opacity:0.6}polygon{stroke:none}polygon.v{fill:#1a1a1a}\
polygon.h{opacity:0.6}circle{stroke:#fff;stroke-width:1.5}";

/// Every figure: its file name in `docs/user/src/figures/` and its SVG text.
///
/// For each cell `NAME` (interval, triangle, quadrilateral, tetrahedron, hexahedron, prism,
/// pyramid) there is `cell-NAME.svg`, the reference cell with its vertex, edge and face
/// numbers, and `lagrange-NAME-pP.svg`, the equispaced Lagrange nodes of degree P numbered in
/// tetraloma's order, for P = 1, 2, 3 on the interval, triangle and quadrilateral and P = 1, 2
/// on the three-dimensional cells.
///
/// # Example
///
/// ```
/// let files = tetraloma_examples::figures::svg_files();
/// assert_eq!(files.len(), 24);
/// for (name, svg) in &files {
///     assert!(name.ends_with(".svg") && svg.starts_with("<svg"));
/// }
/// assert!(files.iter().any(|(name, _)| name == "lagrange-tetrahedron-p2.svg"));
/// ```
pub fn svg_files() -> Vec<(String, String)> {
    let mut files = Vec::new();
    for (cell, name) in CELLS {
        files.push((format!("cell-{name}.svg"), reference_cell_svg(cell, name)));
        let max_degree = if tdim(cell) == 3 { 2 } else { 3 };
        for degree in 1..=max_degree {
            files.push((
                format!("lagrange-{name}-p{degree}.svg"),
                lagrange_svg(cell, name, degree),
            ));
        }
    }
    files
}

/// The reference cell `cell` with its vertex, edge and face numbers and its edge directions.
fn reference_cell_svg(cell: CellType, name: &str) -> String {
    let td = tdim(cell);
    let verts = vertices(cell);
    let visibility = Visibility::new(cell);
    let mut svg = Svg::new(
        cell,
        &format!("The reference {name} with its vertex, edge and face numbers"),
    );
    let edges = sub_entities(cell, 1);
    // Hidden edges first, so that visible ones are drawn over them.
    for front in [false, true] {
        for edge in edges {
            let visible = visibility.visible(&centroid(&[verts[edge[0]], verts[edge[1]]]));
            if visible == front {
                let (a, b) = (svg.px(verts[edge[0]]), svg.px(verts[edge[1]]));
                svg.edge(a, b, !visible, "");
                svg.arrow(a, b, !visible);
            }
        }
    }
    for (i, edge) in edges.iter().enumerate() {
        let mid = centroid(&[verts[edge[0]], verts[edge[1]]]);
        svg.label(svg.px(&mid), &i.to_string(), 1, !visibility.visible(&mid));
    }
    if td >= 2 {
        for (i, face) in sub_entities(cell, 2).iter().enumerate() {
            let corners: Vec<&[f64]> = face.iter().map(|&v| verts[v]).collect();
            let mid = centroid(&corners);
            svg.label(svg.px(&mid), &i.to_string(), 2, !visibility.visible(&mid));
        }
    }
    for (i, &v) in verts.iter().enumerate() {
        let at = svg.px(v);
        let hidden = !visibility.visible(v);
        svg.dot(at, 0, hidden);
        // The label goes where it is furthest from the vertex's edges.
        let directions: Vec<[f64; 2]> = edges
            .iter()
            .filter(|edge| edge.contains(&i))
            .map(|edge| {
                let other = if edge[0] == i { edge[1] } else { edge[0] };
                outward(at, svg.px(verts[other]))
            })
            .collect();
        let away = label_direction(&directions);
        svg.label(
            [
                at[0] + VERTEX_LABEL_OFFSET * away[0],
                at[1] + VERTEX_LABEL_OFFSET * away[1],
            ],
            &i.to_string(),
            0,
            hidden,
        );
    }
    let roles: &[usize] = if td >= 2 { &[0, 1, 2] } else { &[0, 1] };
    svg.legend(roles, td);
    svg.finish()
}

/// The equispaced Lagrange nodes of degree `degree` on `cell`, numbered in tetraloma's order
/// and coloured by the kind of sub-entity they belong to.
fn lagrange_svg(cell: CellType, name: &str, degree: u32) -> String {
    let td = tdim(cell);
    let verts = vertices(cell);
    let visibility = Visibility::new(cell);
    let points = lagrange_points(cell, degree, NodeFamily::Equispaced)
        .expect("equispaced points exist for every cell and degree ≥ 1");
    // The role of every node: the dimension of its sub-entity, or 3 inside the cell.
    let mut roles = vec![0; points.len() / td];
    for dim in 0..=td {
        let role = if dim == td { 3 } else { dim };
        for index in 0..num_sub_entities(cell, dim) {
            for node in lagrange_entity_nodes(cell, degree, dim, index) {
                roles[node] = role;
            }
        }
    }
    let mut svg = Svg::new(
        cell,
        &format!(
            "The Lagrange nodes of degree {degree} on the reference {name}, in tetraloma's order"
        ),
    );
    for front in [false, true] {
        for edge in sub_entities(cell, 1) {
            let visible = visibility.visible(&centroid(&[verts[edge[0]], verts[edge[1]]]));
            if visible == front {
                let (a, b) = (svg.px(verts[edge[0]]), svg.px(verts[edge[1]]));
                svg.edge(a, b, !visible, "l");
            }
        }
    }
    for (node, x) in points.chunks_exact(td).enumerate() {
        let at = svg.px(x);
        let hidden = !visibility.visible(x);
        svg.dot(at, roles[node], hidden);
        svg.node_label(at, node, roles[node], hidden);
    }
    let mut present = roles;
    present.sort_unstable();
    present.dedup();
    svg.legend(&present, td);
    svg.finish()
}

/// The average of `points`.
fn centroid(points: &[&[f64]]) -> Vec<f64> {
    let n = points.len() as f64;
    (0..points[0].len())
        .map(|k| points.iter().map(|p| p[k]).sum::<f64>() / n)
        .collect()
}

/// Of eight compass directions in pixels (y downwards), the one whose largest cosine with
/// `directions` is smallest; the first such in clockwise order from straight up.
fn label_direction(directions: &[[f64; 2]]) -> [f64; 2] {
    let d = core::f64::consts::FRAC_1_SQRT_2;
    let candidates = [
        [0.0, -1.0],
        [d, -d],
        [1.0, 0.0],
        [d, d],
        [0.0, 1.0],
        [-d, d],
        [-1.0, 0.0],
        [-d, -d],
    ];
    let mut best = candidates[0];
    let mut best_cosine = f64::INFINITY;
    for c in candidates {
        let cosine = directions
            .iter()
            .map(|u| c[0] * u[0] + c[1] * u[1])
            .fold(f64::NEG_INFINITY, f64::max);
        if cosine < best_cosine {
            best = c;
            best_cosine = cosine;
        }
    }
    best
}

/// The unit vector from `from` to `to` in pixels; straight up if they coincide.
fn outward(from: [f64; 2], to: [f64; 2]) -> [f64; 2] {
    let d = [to[0] - from[0], to[1] - from[1]];
    let length = (d[0] * d[0] + d[1] * d[1]).sqrt();
    if length < 1e-9 {
        [0.0, -1.0]
    } else {
        [d[0] / length, d[1] / length]
    }
}

/// The oblique projection of a reference point with one to three coordinates to the drawing
/// plane, in reference units with y upwards.
fn project(x: &[f64]) -> [f64; 2] {
    match *x {
        [x0] => [x0, 0.0],
        [x0, x1] => [x0, x1],
        [x0, x1, x2] => [x0 + DEPTH_X * x1, x2 + DEPTH_Y * x1],
        _ => panic!(
            "reference points have one to three coordinates, not {}",
            x.len()
        ),
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn point3(x: &[f64]) -> [f64; 3] {
    [x[0], x[1], x[2]]
}

/// Which faces of a three-dimensional cell face the viewer.
///
/// A point is drawn as visible when it lies on a face that faces the viewer. Since reference
/// cells are convex, this holds for an edge exactly when its midpoint does, and for a face
/// exactly when its centroid does. Every point of a cell of dimension below 3 is visible.
struct Visibility {
    /// The planes of the faces that face the viewer: outward normal n and offset n · x.
    front: Vec<([f64; 3], f64)>,
    /// Whether the cell is three-dimensional.
    solid: bool,
}

impl Visibility {
    fn new(cell: CellType) -> Visibility {
        if tdim(cell) < 3 {
            return Visibility {
                front: Vec::new(),
                solid: false,
            };
        }
        let verts = vertices(cell);
        let centre = point3(&centroid(verts));
        let mut front = Vec::new();
        for face in sub_entities(cell, 2) {
            let p = |k: usize| point3(verts[face[k]]);
            let corners: Vec<&[f64]> = face.iter().map(|&v| verts[v]).collect();
            let mid = point3(&centroid(&corners));
            let mut normal = cross(sub(p(1), p(0)), sub(p(2), p(0)));
            if dot(normal, sub(mid, centre)) < 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
            }
            if dot(normal, VIEW) < 0.0 {
                front.push((normal, dot(normal, p(0))));
            }
        }
        Visibility { front, solid: true }
    }

    fn visible(&self, x: &[f64]) -> bool {
        if !self.solid {
            return true;
        }
        let x = point3(x);
        self.front
            .iter()
            .any(|&(n, offset)| (dot(n, x) - offset).abs() <= 1e-9 * dot(n, n).sqrt())
    }
}

/// An SVG document under construction: the drawing of one cell, scaled to pixels.
struct Svg {
    out: String,
    /// The reference x drawn at the left edge of the drawing area.
    left: f64,
    /// The reference height drawn at the top edge of the drawing area.
    top: f64,
    width: f64,
    height: f64,
}

impl Svg {
    /// A figure of `cell` with the accessible title `title`, sized to the cell's projection.
    fn new(cell: CellType, title: &str) -> Svg {
        let projected: Vec<[f64; 2]> = vertices(cell).iter().map(|&v| project(v)).collect();
        let mut low = [f64::INFINITY; 2];
        let mut high = [f64::NEG_INFINITY; 2];
        for p in &projected {
            low = [low[0].min(p[0]), low[1].min(p[1])];
            high = [high[0].max(p[0]), high[1].max(p[1])];
        }
        let content = (high[0] - low[0]) * SCALE + 2.0 * MARGIN;
        let width = content.max(MIN_WIDTH);
        let height = (high[1] - low[1]) * SCALE + 2.0 * MARGIN + LEGEND_HEIGHT;
        let mut out = String::new();
        out.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" height=\"{height:.0}\" \
             viewBox=\"0 0 {width:.0} {height:.0}\" role=\"img\">\n"
        ));
        out.push_str(&format!("<title>{title}</title>\n"));
        out.push_str(&format!("<style>{STYLE}</style>\n"));
        out.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#fff\"/>\n");
        Svg {
            out,
            left: low[0] - (width - content) / 2.0 / SCALE,
            top: high[1],
            width,
            height,
        }
    }

    /// The pixel position of a reference point.
    fn px(&self, x: &[f64]) -> [f64; 2] {
        let p = project(x);
        [
            MARGIN + (p[0] - self.left) * SCALE,
            MARGIN + (self.top - p[1]) * SCALE,
        ]
    }

    /// The class attribute of `classes` (space-separated, possibly empty) and of a hidden part.
    fn class(classes: &str, hidden: bool) -> String {
        match (classes.is_empty(), hidden) {
            (true, false) => String::new(),
            (true, true) => " class=\"h\"".to_string(),
            (false, false) => format!(" class=\"{classes}\""),
            (false, true) => format!(" class=\"{classes} h\""),
        }
    }

    /// A cell edge from `a` to `b`, dashed if hidden.
    fn edge(&mut self, a: [f64; 2], b: [f64; 2], hidden: bool, classes: &str) {
        self.out.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"{}/>\n",
            a[0],
            a[1],
            b[0],
            b[1],
            Svg::class(classes, hidden)
        ));
    }

    /// An arrowhead three quarters of the way from `a` to `b`, pointing towards `b`: the
    /// direction of the edge.
    fn arrow(&mut self, a: [f64; 2], b: [f64; 2], hidden: bool) {
        let d = outward(a, b);
        let tip = [a[0] + 0.75 * (b[0] - a[0]), a[1] + 0.75 * (b[1] - a[1])];
        let (length, half) = (11.0, 4.5);
        let base = [tip[0] - length * d[0], tip[1] - length * d[1]];
        let wing = [-d[1] * half, d[0] * half];
        self.out.push_str(&format!(
            "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\"{}/>\n",
            tip[0],
            tip[1],
            base[0] + wing[0],
            base[1] + wing[1],
            base[0] - wing[0],
            base[1] - wing[1],
            Svg::class("v", hidden)
        ));
    }

    /// A node or vertex dot in the colour of `role`.
    fn dot(&mut self, at: [f64; 2], role: usize, hidden: bool) {
        self.out.push_str(&format!(
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\"{}/>\n",
            at[0],
            at[1],
            Svg::class(ROLES[role].0, hidden)
        ));
    }

    /// A centred label in the colour of `role`.
    fn label(&mut self, at: [f64; 2], text: &str, role: usize, hidden: bool) {
        self.out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" dy=\"0.35em\"{}>{text}</text>\n",
            at[0],
            at[1],
            Svg::class(ROLES[role].0, hidden)
        ));
    }

    /// The number of a Lagrange node, above and to the right of its dot.
    fn node_label(&mut self, at: [f64; 2], node: usize, role: usize, hidden: bool) {
        self.out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\"{}>{node}</text>\n",
            at[0] + 6.0,
            at[1] - 7.0,
            Svg::class(&format!("n {}", ROLES[role].0), hidden)
        ));
    }

    /// The legend: a dot and a name for each of `roles`, and the coordinate axes of a cell of
    /// dimension `td` in the projection of the figure.
    fn legend(&mut self, roles: &[usize], td: usize) {
        let y = self.height - LEGEND_HEIGHT / 2.0;
        let mut x = 20.0;
        for &role in roles {
            let (class, name) = ROLES[role];
            self.out.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{y:.1}\" r=\"5\" class=\"{class}\"/>\n",
                x + 5.0
            ));
            self.out.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{y:.1}\" dy=\"0.35em\" class=\"k\">{name}</text>\n",
                x + 15.0
            ));
            x += 15.0 + 9.0 * name.len() as f64 + 18.0;
        }
        // The axes: the unit vectors of the reference coordinates, 30 pixels long.
        let origin = [self.width - 80.0, self.height - 12.0];
        let names = ["x", "y", "z"];
        for (k, name) in names.iter().enumerate().take(td) {
            let mut unit = vec![0.0; td];
            unit[k] = 1.0;
            let p = project(&unit);
            let tip = [origin[0] + 30.0 * p[0], origin[1] - 30.0 * p[1]];
            self.out.push_str(&format!(
                "<path class=\"a\" d=\"M{:.1},{:.1}L{:.1},{:.1}\"/>\n",
                origin[0], origin[1], tip[0], tip[1]
            ));
            let d = outward(origin, tip);
            self.out.push_str(&format!(
                "<text x=\"{:.1}\" y=\"{:.1}\" dy=\"0.35em\" class=\"k\" \
                 style=\"text-anchor:middle\">{name}</text>\n",
                tip[0] + 9.0 * d[0],
                tip[1] + 9.0 * d[1]
            ));
        }
    }

    fn finish(mut self) -> String {
        self.out.push_str("</svg>\n");
        self.out
    }
}
