# Feature Dependency Graph & Pipeline Flow Diagram — Feature Engineering v2

## Pipeline Flow Diagram

Execution order through `feature_engineering_v2.py`. Each stage reads only raw
dataset columns plus outputs of *earlier* stages (no forward references),
which is what makes the pipeline a valid single-pass computation.

```mermaid
flowchart TD
    A[Load Enterprise Dataset v2\naccess_logs_v2.csv + entities_v2.csv] --> B[Sort by entity_id, timestamp]
    B --> C[1. User Behavior Features]
    C --> D[2. Device Trust Features]
    D --> E[3. Network Features]
    E --> F[4. Authentication Features]
    F --> G[5. Resource Access Features]
    G --> H[6. Command Sequence Features]
    H --> I[7. Session Features]
    I --> J[8. Organization Features]
    J --> K[9. Behavioral Baseline Features]
    K --> L[10. Temporal Features]
    L --> M[11. Attack-Specific Composite Features]
    M --> N[Feature Selection\ndrop 21 scratch/intermediate columns]
    N --> O[Feature Validation\nfill cold-start NaNs, clip infs]
    O --> P[Memory Optimization\ndowncast dtypes, 18.2% reduction]
    P --> Q[Export: features_v2.csv\n86 curated features]
```

## Feature Dependency Graph

Which categories build on which. Arrows mean "reads output columns from."

```mermaid
flowchart LR
    RAW[Raw Dataset v2 Columns] --> UB[User Behavior]
    RAW --> DT[Device Trust]
    RAW --> NET[Network]
    RAW --> AUTH[Authentication]
    RAW --> RES[Resource Access]
    RAW --> CMD[Command Sequence]
    RAW --> SESS[Session]

    UB --> ORG[Organization]
    RES --> ORG
    RES --> TEMP[Temporal]
    UB --> TEMP
    UB --> BASE[Behavioral Baseline]

    DT --> ATTACK[Attack-Specific Composites]
    NET --> ATTACK
    AUTH --> ATTACK
    RES --> ATTACK
    CMD --> ATTACK
    SESS --> ATTACK
    ORG --> ATTACK
    TEMP --> ATTACK
    BASE --> ATTACK

    ATTACK --> OUT[features_v2.csv]
```

## Why This Order Matters

1. **User Behavior, Device Trust, Network, Authentication, Resource Access, Command Sequence, Session** are all "leaf" categories — they read only raw dataset columns (plus their own shift/rolling history). They can technically run in any order relative to each other.
2. **Organization** depends on Resource Access (`resource_sensitivity`) and general entity attributes — must run after Resource Access.
3. **Behavioral Baseline** and **Temporal** depend on User Behavior's `login_hour`/`expanding_mean_hour` computations.
4. **Attack-Specific Composites** is deliberately LAST — every composite score (e.g. `insider_threat_score`) is a weighted blend of columns from multiple earlier categories (`resource_sensitivity_deviation` from Resource Access + `peer_group_resource_sensitivity_deviation` from Organization), so it cannot run until all its inputs exist.
5. **Feature Selection → Validation → Memory Optimization → Export** always runs last, once, over the fully-assembled feature matrix.

## Key Design Note: No Circular Dependencies

Every stage function takes `df` and returns `df` with *new* columns appended — no stage ever needs to re-read or modify a column produced by a *later* stage. This makes the pipeline trivially parallelizable across independent leaf categories in a future Spark/Flink port, and makes debugging tractable (a bug in Stage N can only be caused by Stage N's own logic or genuine data issues in Stages 1..N-1, never by anything downstream).
