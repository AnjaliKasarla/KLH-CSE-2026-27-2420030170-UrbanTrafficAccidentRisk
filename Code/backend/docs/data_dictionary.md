# \# Data Dictionary

# 

# \## 1. Dataset Overview

# 

# \*\*Dataset:\*\* Road Accident Dataset

# 

# \*\*Source file:\*\*

# 

# `data/raw/road\_accident/Road Accident Data.csv`

# 

# \*\*Records:\*\* 307,973

# 

# \*\*Features:\*\* 23

# 

# \*\*Target:\*\* `Accident\_Severity`

# 

# The dataset contains structured accident records covering temporal,

# road, junction, environmental, geographic, vehicle, casualty, and

# traffic-related attributes.

# 

# \---

# 

# \## 2. Column Definitions

# 

# | Column | Data Type | Role | Missing Values | Description |

# |---|---|---|---:|---|

# | `Accident\_Index` | string | Identifier | 0 | Unique accident record identifier. |

# | `Accident Date` | string | Temporal | 0 | Date on which the accident was recorded. |

# | `Month` | string | Temporal | 0 | Month associated with the accident. |

# | `Day\_of\_Week` | string | Temporal | 0 | Day of the week associated with the accident. |

# | `Year` | integer | Temporal | 0 | Year associated with the accident. |

# | `Junction\_Control` | string | Road/Junction | 0 | Type of control associated with the junction. |

# | `Junction\_Detail` | string | Road/Junction | 0 | Detailed junction configuration. |

# | `Accident\_Severity` | string | \*\*Target\*\* | 0 | Accident severity class: Slight, Serious, or Fatal. |

# | `Latitude` | float | Location | 0 | Geographic latitude of the accident location. |

# | `Light\_Conditions` | string | Environmental | 0 | Lighting conditions at the time of the accident. |

# | `Local\_Authority\_(District)` | string | Location | 0 | Local authority/district associated with the accident. |

# | `Carriageway\_Hazards` | string | Road Condition | 302,549 | Hazard information associated with the carriageway. |

# | `Longitude` | float | Location | 0 | Geographic longitude of the accident location. |

# | `Number\_of\_Casualties` | integer | Accident Impact | 0 | Number of casualties associated with the accident. |

# | `Number\_of\_Vehicles` | integer | Accident Context | 0 | Number of vehicles involved in the accident. |

# | `Police\_Force` | string | Administrative | 0 | Police force associated with the accident record. |

# | `Road\_Surface\_Conditions` | string | Road Condition | 317 | Surface condition of the road. |

# | `Road\_Type` | string | Road Context | 1,534 | Type of road on which the accident occurred. |

# | `Speed\_limit` | integer | Road Context | 0 | Speed limit associated with the accident location. |

# | `Time` | string | Temporal | 17 | Time associated with the accident. |

# | `Urban\_or\_Rural\_Area` | string | Location Context | 0 | Indicates whether the accident occurred in an urban or rural area. |

# | `Weather\_Conditions` | string | Environmental | 6,057 | Weather conditions associated with the accident. |

# | `Vehicle\_Type` | string | Vehicle Context | 0 | Type of vehicle involved in the accident record. |

# 

# \---

# 

# \## 3. Feature Groups

# 

# \### Temporal Features

# 

# \- `Accident Date`

# \- `Month`

# \- `Day\_of\_Week`

# \- `Year`

# \- `Time`

# 

# These features will support temporal pattern analysis and derived

# features such as hour, day period, weekday/weekend, and seasonal

# patterns.

# 

# \### Geographic Features

# 

# \- `Latitude`

# \- `Longitude`

# \- `Local\_Authority\_(District)`

# \- `Urban\_or\_Rural\_Area`

# 

# These features provide geographic and urbanization context.

# 

# \### Road and Junction Features

# 

# \- `Junction\_Control`

# \- `Junction\_Detail`

# \- `Road\_Type`

# \- `Road\_Surface\_Conditions`

# \- `Carriageway\_Hazards`

# \- `Speed\_limit`

# 

# These features represent road infrastructure and road-condition

# information.

# 

# \### Environmental Features

# 

# \- `Light\_Conditions`

# \- `Weather\_Conditions`

# 

# These features represent environmental conditions surrounding the

# accident.

# 

# \### Vehicle and Accident Context

# 

# \- `Vehicle\_Type`

# \- `Number\_of\_Vehicles`

# \- `Number\_of\_Casualties`

# 

# These features describe the vehicles involved and accident impact.

# 

# \### Administrative Features

# 

# \- `Police\_Force`

# \- `Local\_Authority\_(District)`

# 

# These features provide administrative/geographical context.

# 

# \### Identifier

# 

# \- `Accident\_Index`

# 

# `Accident\_Index` is treated as an identifier rather than a predictive

# feature.

# 

# \---

# 

# \## 4. Missing-Value Summary

# 

# The observed missing values are:

# 

# | Column | Missing Count | Missing Percentage |

# |---|---:|---:|

# | `Carriageway\_Hazards` | 302,549 | 98.24% |

# | `Weather\_Conditions` | 6,057 | 1.97% |

# | `Road\_Type` | 1,534 | 0.50% |

# | `Road\_Surface\_Conditions` | 317 | 0.10% |

# | `Time` | 17 | 0.01% |

# 

# Missing-value handling will be performed during preprocessing rather

# than modifying the raw dataset.

# 

# \---

# 

# \## 5. Data Quality Observations

# 

# The initial profiling identified:

# 

# \- 307,973 records.

# \- 23 columns.

# \- 1 duplicate row.

# \- No missing values in the target column.

# \- The target contains three severity classes.

# \- `Carriageway\_Hazards` has a very high missing-value rate.

# \- `Weather\_Conditions`, `Road\_Type`, `Road\_Surface\_Conditions`, and

# &#x20; `Time` contain smaller amounts of missing data.

# 

# These observations will be considered during Phase 1 preprocessing

# and feature engineering.

# 

# \---

# 

# \## 6. Modeling Considerations

# 

# The raw dataset will remain unchanged.

# 

# Preprocessing will later address:

# 

# 1\. Duplicate records.

# 2\. Missing values.

# 3\. Date and time conversion.

# 4\. Categorical encoding.

# 5\. Temporal feature extraction.

# 6\. Feature scaling where required.

# 7\. Identifier exclusion from predictive modeling.

# 8\. Class imbalance in `Accident\_Severity`.

# 

# All preprocessing transformations should be fitted using the training

# data and applied consistently to validation/test data.

