//! `EntityMap`: the map from submesh entities to parent entities. An opaque placeholder until
//! M9, which fills it (ADR-0012).

/// The map from the cells of a submesh to the entities of its parent. Opaque until M9.
#[derive(Debug, Default)]
pub struct EntityMap {
    _private: (),
}
