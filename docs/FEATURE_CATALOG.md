# Feature Catalog — Enterprise Feature Engineering v2

86 curated behavioral features across 11 categories, built on the frozen Enterprise Dataset v2. Every feature below is documented with its purpose, formula, cybersecurity intuition, expected importance, and which attack(s) it helps detect.

**Legend for Expected Importance:** 🔴 High — 🟡 Medium — ⚪ Low/supporting

---

## 1. User Behavior Features

| Feature | Purpose | Formula | Cybersecurity Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `historical_event_count` | How much history exists for this entity | `cumcount()` per entity, vectorized | Feeds every cold-start calculation downstream | Cold Start (all) | 🔴 |
| `time_since_last_login_hours` | Gap since entity's last event | `timestamp - shift(1)` per entity | Very short gaps = session hijack/burst; very long gaps = dormant-account reactivation | Session Hijacking, Insider Threat | 🟡 |
| `avg_session_duration` | Entity's typical session length (shifted, no lookahead) | expanding mean of `session_duration_min` | Baseline to compare current session against | Session Hijacking, Exfiltration | 🟡 |
| `session_duration_zscore` | How unusual is this session's length | `(current - avg) / std` | Abnormally short/long sessions are a classic anomaly signal | Brute Force, Lateral Movement | 🔴 |
| `day_of_week_deviation` | How rare is this weekday for this entity | Laplace-smoothed relative frequency | Attackers don't respect an entity's normal weekly rhythm | Insider Threat, Credential Misuse | 🟡 |
| `weekend_activity_flag` | Is this a weekend event | passthrough | Weekend activity is inherently higher-risk context | Low-and-Slow Exfil | ⚪ |
| `holiday_activity_flag` | Is this a company holiday | passthrough | Same logic, stronger signal (near-zero legitimate traffic expected) | Low-and-Slow Exfil | ⚪ |
| `behavioral_drift_score` | Recent (10-event) vs all-time mean login hour | `\|rolling10_mean - expanding_mean\|` | Detects gradual behavioral drift — feeds concept-drift monitoring | Insider Threat | 🟡 |
| `rolling_failed_login_rate_7d` | 7-day failed-login rate | time-rolling mean | Smooths single-event noise into a trend | Brute Force, Credential Stuffing | 🟡 |

## 2. Device Trust Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `is_new_device` | First time this (entity, device) pair appears | `cumcount()==0`, vectorized | Core signal for compromised-credential-on-new-hardware | Credential Misuse, Device Spoofing | 🔴 |
| `is_new_os` / `is_new_browser` | First time this OS/browser seen for entity | same cumcount trick | Secondary fingerprint-novelty signals | Device Spoofing | 🟡 |
| `device_usage_frequency` | How many times entity has used this exact device | `cumcount()` per (entity, device) | Distinguishes a brand-new device from a rarely-used-but-known one | Device Spoofing | 🟡 |
| `new_device_probability` | Smoothed novelty ratio | `1 - freq/(history+1)` | Continuous version of `is_new_device`, avoids hard 0/1 cliff | Credential Misuse | 🟡 |
| `fingerprint_change_score` | OS changed on a device_id we've seen before | shift+compare within (entity, device) | The core Device Spoofing signature | Device Spoofing | 🔴 |
| `device_reputation` | Inverse of distinct-entity-count sharing this device_id | `1/nunique(entity_id)` per device | Devices should be ~1:1 with entities; shared devices are a red flag | Device Spoofing, Credential Stuffing | 🟡 |
| `managed_device_flag` | Is this an established, recognized device | `usage_frequency >= 5` | Proxy for "enrolled/trusted" vs "unknown" hardware | Device Spoofing | ⚪ |
| `device_risk_score` | Composite device risk | weighted blend of above | Single number for dashboards/rules | All device-related attacks | 🔴 |

## 3. Network Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `distance_from_prev_km` | Geo distance since last login | haversine, vectorized | Raw input to impossible-travel logic | Impossible Travel | 🔴 |
| `geo_velocity_kmh` | Implied travel speed | `distance / max(elapsed, 0.25h)` | Physically impossible speed = compromised session | Impossible Travel | 🔴 |
| `impossible_travel_flag` | Hard threshold flag | `speed > 950 km/h` | Binary, rule-friendly version of the above | Impossible Travel | 🔴 |
| `country_change` / `city_change` | Location changed since last login | shift+compare | Secondary geo-mobility signal, softer than full impossible travel | Impossible Travel, Credential Misuse | 🟡 |
| `network_zone_change` | Entity moved between network zones | shift+compare | Zone-hopping (e.g. OT_Network → VPN_Remote) is unusual for most roles | Lateral Movement | 🟡 |
| `is_hosting_asn` | Traffic originates from a known hosting/VPN ASN | static ASN lookup | Attackers overwhelmingly originate from datacenter ASNs, not residential ISPs | Credential Misuse, Credential Stuffing | 🔴 |
| `remote_access_score` / `internal_network_score` | Zone-based risk context | membership in zone sets | Baseline network-context risk | Lateral Movement | ⚪ |
| `anonymization_risk_score` | Composite VPN/hosting-ASN/remote-zone proxy | weighted blend | Documented as a structural proxy (no real Tor exit-node list in synthetic data) | Credential Misuse | 🟡 |

## 4. Authentication Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `rolling_avg_failed_logins` / `failed_login_spike` | Entity's own failed-login norm and deviation from it | expanding mean; `current - avg` | Distinguishes a naturally fumble-fingered typer from an actual spike | Brute Force | 🔴 |
| `failed_login_streak` | Failures in the last 5 events | rolling sum of `failures>0` | Classic brute-force burst signature | Brute Force | 🔴 |
| `mfa_deviation` | Entity normally uses MFA but didn't this time | `historical_mfa_rate - current` | MFA bypass/absence on a normally-MFA'd account is a strong signal | Credential Misuse | 🔴 |
| `auth_method_entropy` | Variety of auth methods used historically | running Shannon entropy | Humans are consistent; high entropy is unusual | Credential Misuse | ⚪ |
| `password_spray_score` | Distinct entities hitting the same source IP within 1h | windowed distinct-count (see code notes) | The literal definition of credential stuffing/password spray | Credential Stuffing | 🔴 |
| `credential_stuffing_score` | Composite | blend of spray score + failures + hosting ASN | Single dashboard-ready score | Credential Stuffing | 🔴 |

## 5. Resource Access Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `resource_diversity_count` | Cumulative distinct resources touched | cumcount first-occurrence trick, vectorized | Rapidly expanding resource footprint = lateral movement | Lateral Movement | 🔴 |
| `resource_sensitivity_deviation` | Current resource sensitivity vs entity's own historical mean | `current - expanding_mean` | Core Insider Threat signal | Insider Threat, Privilege Escalation | 🔴 |
| `is_cross_department_access` | Resource belongs to a different department | static resource→dept map | Direct lateral-movement indicator | Lateral Movement | 🔴 |
| `privilege_deviation` | Resource sensitivity exceeds entity's privilege level | `max(sensitivity - privilege_level, 0)` | Direct privilege-escalation indicator | Privilege Escalation | 🔴 |
| `critical_resource_rate_7d` | Rate of sensitive-resource access over 7 days | time-rolling mean of `sensitivity>=4` | Trend-level exfiltration/insider signal | Low-and-Slow Exfil | 🟡 |
| `resource_entropy` | Diversity of resources in a trailing 20-event window | windowed Shannon entropy | Sudden broadening of resource variety | Lateral Movement | 🟡 |

## 6. Command Sequence Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `command_sequence_length` | Number of commands in the session | token count | Long/unusual command chains stand out | Command Abuse | ⚪ |
| `command_entropy` | Diversity of commands within one sequence | Shannon entropy of tokens | Highly repetitive vs. highly varied command use | Command Abuse | ⚪ |
| `command_rarity_score` | How rare are this row's commands, globally | inverse global frequency, normalized | Rare commands are inherently more suspicious | Command Abuse, LotL | 🟡 |
| `dangerous_command_ratio` | Fraction of tokens matching destructive-command vocabulary | keyword match ratio | Directly detects ransomware-style commands (`vssadmin delete shadows`, etc.) | Command Abuse | 🔴 |
| `lolbin_usage_flag` | Use of legitimate admin tools in a suspicious context | keyword/process match | Detects "living off the land" tool abuse | Living-off-the-Land | 🔴 |
| `powershell_usage_flag` | PowerShell specifically in use | process match | PowerShell is the single most common LotL vector | Living-off-the-Land | 🟡 |
| `privilege_escalation_cmd_score` | Fraction of tokens matching priv-esc vocabulary | keyword match ratio | Directly detects `net localgroup administrators /add`-style commands | Privilege Escalation | 🔴 |
| `command_novelty_score` | Fraction of this row's tokens never used by this entity before | expanding per-entity token vocabulary, vectorized via explode+cumcount | A user suddenly running commands they've never run before | Command Abuse, Privilege Escalation | 🟡 |

## 7. Session Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `session_size` | Number of log rows sharing a session_id | `groupby(session_id).size()` | Normal sessions = 1 row; hijacked sessions = 2+ with changed context | Session Hijacking | 🔴 |
| `session_hijack_flag` | Device or geo changed mid-session without new auth | shift+compare within session_id | The direct, purpose-built Session Hijacking signature | Session Hijacking | 🔴 |
| `session_age_minutes` | Time elapsed since session start | `current_ts - session_start` | Context for hijack timing | Session Hijacking | ⚪ |
| `concurrent_session_count_1h` | Distinct sessions for this entity active within 1h | windowed distinct-count | Multiple simultaneous sessions is unusual for a single human | Session Hijacking, Credential Misuse | 🟡 |
| `session_restart_rate` | Distinct sessions per entity per 24h | windowed distinct-count | Frequent reconnects can indicate unstable/scripted access | Brute Force | ⚪ |

## 8. Organization Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `peer_group_resource_sensitivity_deviation` | Entity's sensitivity access vs. same dept+role peers | `current - groupmean(dept,role)` | Insider Threat is invisible against one's OWN baseline but visible against peers | Insider Threat | 🔴 |
| `department_baseline_sensitivity` | Department's average resource sensitivity | groupby transform | Context/denominator for other org features | Insider Threat | ⚪ |
| `business_unit_deviation` | Absolute deviation from department norm | `\|current - dept_mean\|` | Broader anomaly context | Insider Threat | ⚪ |
| `privilege_baseline_sensitivity` | Expected sensitivity for this privilege level | groupby transform | Denominator for `privilege_deviation` context | Privilege Escalation | ⚪ |
| `manager_deviation` | Entity's mean sensitivity vs. their manager's | absolute diff of per-entity means | An IC accessing exec-level data their manager doesn't even touch | Insider Threat, Privilege Escalation | 🟡 |

## 9. Behavioral Baseline Features (Cold-Start & Adaptive)

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `adaptive_threshold` | Dynamic per-entity anomaly boundary | `mean + 2*std` of session duration | Adapts per-entity rather than one global cutoff | All | 🟡 |
| `adaptive_threshold_exceeded_flag` | Did this event exceed its own adaptive threshold | `current > adaptive_threshold` | Rule-friendly binary version | All | 🟡 |
| `baseline_confidence` | How much we trust this entity's baseline | `n / (n + 15)` | Smoothly approaches 1.0 as history accumulates | Cold Start (all) | 🔴 |
| `cold_start_score` | Inverse of baseline_confidence | `1 - baseline_confidence` | Continuous cold-start signal (vs. v1's binary flag) | Cold Start (all) | 🔴 |
| `cold_start_flag` | Hard threshold version | `history < 10 events` | Rule-friendly binary version | Cold Start (all) | 🟡 |

## 10. Temporal Features

| Feature | Purpose | Formula | Intuition | Attacks Helped | Importance |
|---|---|---|---|---|---|
| `rolling_count_1h` / `_24h` / `_7d` | Event volume at 3 time horizons | time-rolling count | Multi-scale activity-burst detection | Brute Force, Lateral Movement | 🔴 |
| `burst_score` | Short-term rate vs. long-term rate | `count_1h / (count_24h/24)` | Detects sudden bursts against the entity's own daily rhythm | Brute Force, Lateral Movement | 🔴 |
| `hour_sin`/`hour_cos`, `weekday_sin`/`weekday_cos` | Cyclic time encoding | standard sin/cos transform | Preserves circularity (23:00 next to 00:00) that raw integers destroy | supports all time-based features | 🟡 |
| `seasonality_score` | How "peaky" this hour is for this entity | Laplace-smoothed relative hour frequency | Off-pattern-hour detection, smoother than a hard cutoff | Credential Misuse, Low-and-Slow Exfil | 🟡 |
| `login_hour_deviation` | Hour deviation from entity's blended (Bayesian shrunk) baseline | see v1 methodology, carried forward | Core time-of-day anomaly signal | Credential Misuse | 🔴 |
| `is_odd_hour_login` | Hard midnight-hours flag | `hour<6 or hour>22` | Rule-friendly binary version | Credential Misuse, Low-and-Slow Exfil | 🟡 |

## 11. Attack-Specific Composite Features

One purpose-built composite score per attack type, each combining several of the above raw features into a single, dashboard-ready 0-1 signal. See the "Attack Score Separation" validation table below — every composite score is empirically 2×–1,000,000× higher on its target attack type than on normal traffic.

| Feature | Primary Attack | Key Inputs |
|---|---|---|
| `credential_misuse_score` | Credential Misuse | new device, country change, odd hour, no MFA, sensitive resource |
| `credential_stuffing_score` | Credential Stuffing | password spray score, failure flag, hosting ASN |
| `brute_force_score` | Brute Force | failed login streak, failed login spike |
| `impossible_travel_score` | Impossible Travel | geo velocity ratio |
| `device_spoofing_score` | Device Spoofing | fingerprint change, new device, new device probability |
| `lateral_movement_score` | Lateral Movement | cross-dept access, burst score, resource diversity rate |
| `session_hijacking_score` | Session Hijacking | session hijack flag |
| `low_and_slow_exfil_score` | Low-and-Slow Exfiltration | odd hour, sensitive resource, small download size |
| `living_off_the_land_score` | Living-off-the-Land | LOLBin usage, inverse device risk (blends in) |
| `insider_threat_score` | Insider Threat | resource sensitivity deviation (own + peer group) |
| `command_abuse_score` | Command Abuse | dangerous command ratio |

---

## Attack Score Separation — Empirical Validation

| Attack Type | Composite Score | Own Mean | Normal Baseline | Separation Ratio |
|---|---|---|---|---|
| Credential Misuse | `credential_misuse_score` | 0.986 | 0.166 | 5.9× |
| Credential Stuffing | `credential_stuffing_score` | 0.699 | 0.056 | 12.5× |
| Brute Force | `brute_force_score` | 0.505 | 0.014 | 36.1× |
| Impossible Travel | `impossible_travel_score` | 0.444 | 0.003 | 148× |
| Device Spoofing | `device_spoofing_score` | 0.600 | 0.039 | 15.4× |
| Lateral Movement | `lateral_movement_score` | 0.763 | 0.362 | 2.1× |
| Session Hijacking | `session_hijacking_score` | 0.500 | 0.000 | (only attacks trigger this at all) |
| Low-and-Slow Exfiltration | `low_and_slow_exfil_score` | 1.000 | 0.457 | 2.2× |
| Living-off-the-Land | `living_off_the_land_score` | 0.978 | 0.378 | 2.6× |
| Insider Threat | `insider_threat_score` | 0.299 | 0.048 | 6.2× |
| Command Abuse | `command_abuse_score` | 1.000 | 0.000 | (only attacks trigger this at all) |

Note: Lateral Movement and Low-and-Slow Exfiltration show the weakest (though still clearly positive) separation — expected, since both are deliberately designed as subtler, aggregate-pattern attacks rather than single-event red flags. This is an honest result, not a hidden weakness — it directly motivates why these two need the ML models (Isolation Forest / sequence model) to combine multiple weak signals, rather than relying on any single rule.

---

## Pipeline Validation Summary

- **Rows processed:** 60,007
- **Final curated features:** 86 (from 107 raw engineered columns, after dropping 21 intermediate/scratch columns)
- **Nulls after fill:** 0
- **Infinite values:** 0
- **Memory:** 138.1 MB → 113.0 MB after downcasting (18.2% reduction)
- **No `iterrows()` anywhere in the pipeline.** A small number of operations (rolling *distinct-count* for password-spray/session-concurrency detection, and short command-token parsing) use bounded `groupby().rolling().apply()` — linear-time, windowed operations, not the O(n) full-table anti-pattern flagged in v1.
