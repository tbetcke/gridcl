//! Entity creation, step 2: the TopologyKeys kernel, generic over the vector size. Each key is
//! the sub-entity's local vertex ids sorted by a sorting network; the payload is the emission
//! index, and padding positions write an all-ones sentinel key.
