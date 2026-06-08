# Attribute Ranking Pipeline
This document tracks the evolution of the attribute ranking approach used to identify the most relevant product attributes within a category.

## Version 1: Semantic-Only Attribute Matching
The first version relied only on semantic similarity between category names and attribute descriptions, using raw text.

### 1. Approach
- Extracted category names and attribute descriptions
- Directly applied a Sentence Transformer model
- Generated embeddings for:
  - Category text
  - Attribute description text
- Ranked attributes using cosine similarity

### 2. Key Characteristics
- No text normalization or cleanup
- No sales, SKU count, or coverage signals
- All attributes treated equally

### 3. Limitations
- Rankings were sensitive to noisy or abbreviated text
- Rare or niche attributes could rank high if text sounded relevant
- No notion of real-world business importance

## Version 2: Text Cleanup + Rule-Based Attribute Filtering
The second version introduced text preprocessing and hard filters before applying semantic ranking.

### 1. Approach
#### 1.1 Text Cleanup and Normalization
Before embedding, basic text cleanup was applied:
- Abbreviation expansion (for example: ORG → organic, FAT → fat)
- Lowercasing and token normalization
- Minimal category name cleanup
This improved embedding quality and reduced noise from encoded or abbreviated attribute names.

#### 1.2 Rule-Based Attribute Filters
Attributes were retained only if:
- ATT_SKU_COUNT ≥ 1% of CATEGORY_SKU_COUNT
- ATT_TOTAL_SPENT > 0

### 2. Key Characteristics
#### 2.1 Intent
- Remove attributes that appear on very few SKUs
- Exclude attributes never associated with selling products
- Reduce noise before running the sentence transformer

#### 2.2 Pipeline Flow
- Apply text cleanup
- Apply hard filters (SKU coverage + sales)
- Run sentence transformer on remaining attributes
- Rank using cosine similarity

### 3. Limitations
#### 3.1 Limitations Observed
- Many relevant-sounding attributes were removed entirely
- Hard thresholds caused over-pruning
- Long-tail but meaningful attributes were lost
- Filtering was binary (keep/drop)

## Version 3: Sales-Weighted Attribute Importance 
Version 3 removes hard filters and introduces continuous sales-based weighting, which is later combined with semantic relevance.

### 1. Approach
#### 1.1 Step 1: Attribute Sales Attribution
Sales are associated with attributes through SKUs.
Process:
- Aggregate total sales at SKU level
- Attach SKU sales to all attributes present on that SKU
- Aggregate sales at:
  - Category × Attribute × Attribute Value
- Sum across all values to get Category × Attribute sales

#### 1.2 Key outputs:
- ATTRIBUTE_TOTAL_SPENT
- ATTRIBUTE_SKU_COUNT

#### 1.3 Step 2: Attribute Weight Calculation
- Each attribute is assigned a continuous weight: ATTRIBUTE_WEIGHT = ATTRIBUTE_TOTAL_SPENT / CATEGORY_TOTAL_SPENT
- Interpretation:
  - Measures share of category sales coming from SKUs that have this attribute
  - Weights do not sum to 1
  - Attributes overlap through shared SKUs

#### 1.4 Step 3: Semantic Ranking
- Category names and cleaned attribute descriptions are embedded
- Cosine similarity produces a semantic relevance score
- No attributes are removed before embedding

#### 1.5 Step 4: Final Ranking Methods
Supported ranking modes:
- Semantic Only
  - Rank by cosine similarity only
- Sales Only
  - Rank by ATTRIBUTE_WEIGHT only
- Combined (Multiplicative MAUT)
  - Combines normalized semantic score and sales weight
  - Attributes must be both relevant and impactful
- Reciprocal Rank Fusion (RRF)
  - Separately rank by semantic relevance and sales weight
  - Combine rankings using reciprocal rank fusion
  - Reduces sensitivity to outliers

### 2. Key Characteristics
#### 2.1 Key Improvements in Version 3
  - No hard filters that drop attributes
  - Sales treated as a continuous importance signal
  - Better balance between language relevance and business impact
  - More stable and interpretable rankings

### 3. Limitations
#### 3.1 Why this approach doesn’t fully align with CDT:
CDT works very differently from sales or relevance ranking. It does not ask which attributes are big or popular. Instead, it focuses on which attributes explain how shoppers switch between products. Sales‑based or weighted approaches emphasize scale and distribution, but CDT learns importance only from substitution behavior observed in household purchases. As a result, an attribute can rank high using sales or combined weighting yet still play a limited role in actual shopper decision‑making within the CDT framework.
Using a semantic only approach can be more appropriate at the attribute selection stage because it focuses purely on category relevance and shopper language. It helps identify attributes that conceptually belong to how shoppers think about the category, without introducing biases from sales performance or assortment structure. This keeps the role of attribute selection aligned with CDT design, where true importance is determined later by observed switching behavior rather than upfront signals.

## Version 4: Domain × Tier Weighted Semantic Ranking
Version 4 introduces domain-aware weighting by classifying categories into business domains and attributes into behavior tiers, then combining these with semantic similarity.

### 1. Approach
#### 1.1 Step 1: Category Domain Assignment
Every category is classified into exactly one domain using deterministic prefix + keyword rules (first-match-wins).
- 10 domains: FOOD_FRESH, FOOD_PACKAGED, FOOD_BEVERAGE, FOOD_SPECIALTY_DIETARY, HEALTH_WELLNESS, HOUSEHOLD_UTILITY, HARDWARE_ELECTRONICS, NONFOOD_DISCRETIONARY, NONFOOD_HOME, UNKNOWN
- Tiered rule system (T0–T7) for classification

#### 1.2 Step 2: Attribute Tier Classification
Each attribute is classified into a behavior tier using keyword/pattern rules on ATTRIBUTE_DESCRIPTION.
- 18 tiers: OPER, META, BRAND, PACK, CLAIM, SRC, PI, BC, FOOD, BEV, HW, HH, HPC, SENS_FOOD, SENS, TARG, NEUT, NUTR_PASSIVE
- NUTR_PASSIVE: Nutritional facts that are ubiquitous on labels but rarely drive purchase decisions

#### 1.3 Step 3: Domain × Tier Priority Mapping
Each domain assigns priority weights to tiers:
- HIGH = 1.0 — primary differentiator for this domain
- MEDIUM = 0.7 — useful supporting signal
- LOW = 0.3 — noise or irrelevant in this domain
- DEFAULT = 0.5 — tier not listed for a domain

#### 1.4 Step 4: Exclusion Rules
Two exclusion steps remove non-discriminative attributes:
- Exclusion 1: Universal attributes (appear in >85% of categories AND cover >80% of SKUs in >85% of those categories)
- Exclusion 2: Single-value (constant) attributes (only 1 distinct value in >80% of categories they appear in)

#### 1.5 Step 5: Semantic Ranking with Domain Weight
Two ranking modes:
- Semantic: SCORE = cosine_similarity(attribute, category)
- Weighted: SCORE = cosine_similarity(attribute, category) × FINAL_WEIGHT (domain-driven)

Uses e5-large-v2 embedding model. No sales weights — importance is derived from domain/tier rules only.

### 2. Key Characteristics
#### 2.1 Key Improvements in Version 4
- Domain-aware weighting tailors attribute importance to category context
- Behavior tier classification captures attribute purpose (sensory, operational, food-specific, etc.)
- Exclusion rules remove universal and constant attributes that cannot differentiate products
- No sales-based weighting — importance is structural rather than revenue-driven

### 3. Limitations
#### 3.1 Limitations Observed
- Domain and tier rules are manually crafted and may not cover all edge cases
- Domain assignment relies on category naming conventions (prefix-based)
- Fixed priority weights (1.0 / 0.7 / 0.3) may not reflect true relative importance
- No validation against actual shopper switching behavior

## Version 5: Domain × Tier Weighted Ranking + LLM Selection
Version 5 extends Version 4 by adding a pre-filter step for minimum sales/SKU coverage and an LLM selection stage that uses a language model to independently judge and select the most relevant attributes.

### 1. Approach
#### 1.1 Step 1: Pre-Filter (Minimum Sales & SKU Coverage)
Before exclusion rules, attributes are filtered:
- ATTRIBUTE_TOTAL_SPENT > 0 — remove zero-sales attributes
- SKU coverage >= 5% of category — attribute must appear on at least 5% of SKUs in its category

#### 1.2 Step 2: Domain Assignment + Tier Classification + Exclusions
Same as Version 4:
- Category domain assignment (10 domains)
- Attribute behavior tier classification (18 tiers)
- Domain × Tier priority weights (HIGH/MEDIUM/LOW/DEFAULT)
- Exclusion 1: Universal attributes
- Exclusion 2: Single-value attributes

#### 1.3 Step 3: Semantic Ranking with Domain Weight
Same modes as Version 4:
- Semantic: pure cosine similarity
- Weighted: cosine similarity × FINAL_WEIGHT

#### 1.4 Step 4: Export Top-50 for LLM Selection
The top 50 embedding-ranked attributes per category are exported with enriched context for LLM processing:
- Category name, attribute ID, attribute description
- Attribute SKU count, attribute total sales
- TOP_VALUES: summary string of top 10 attribute values (value name, sales, SKU count)
- RANK and SCORE are intentionally excluded from the export to avoid biasing the LLM's independent judgment

#### 1.5 LLM Input/Output
- Inputs: Category-wise CSV files generated from the top-N export for LLM selection
- Outputs: Category-wise CSV files containing LLM-selected attribute results

### 2. Key Characteristics
#### 2.1 Key Improvements in Version 5
- Pre-filter removes zero-sales and very low coverage attributes early, reducing noise
- LLM selection provides an independent, context-aware judgment of attribute relevance
- Enriched export includes value-level detail for LLM to reason about attribute diversity
- Two-stage approach: embedding narrows candidates, LLM makes final selection

### 3. Limitations
#### 3.1 Limitations Observed
- LLM selection depends on prompt design and model capabilities
- Additional cost and latency from LLM inference
- LLM may not have domain-specific retail/CDT knowledge without careful prompting
- Even with temperature=0, the model may produce slightly different selections across runs, making results not fully deterministic
