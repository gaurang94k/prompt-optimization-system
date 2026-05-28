# SQL E-Commerce Analytics

Business analytics on the Olist Brazilian E-Commerce dataset, demonstrating SQL proficiency through real business questions.

## Business Questions Answered

| # | Question | SQL Concepts |
|---|----------|-------------|
| 1 | Setup & first queries | SELECT, WHERE, GROUP BY, JOIN |
| 2 | Revenue analysis & trends | Aggregations, date functions, MoM growth |
| 3 | Customer cohort retention | CTEs, window functions, cohort analysis |
| 4 | Delivery & review analysis | CASE, conditional aggregation, correlation |
| 5 | Advanced: Pareto, RFM, repeat rates | Subqueries, percentiles, business frameworks |

## Tech Stack

- **SQL engine:** DuckDB (portable, zero-config — same SQL syntax as PostgreSQL)
- **Environment:** Jupyter Notebook (Python + SQL)
- **Data:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)
- **Visualization:** matplotlib / pandas plotting

## Setup

```bash
pip install duckdb pandas matplotlib jupyter
cd notebooks/
jupyter notebook
```

## Dataset Schema

```
customers ──┐
             ├── orders ── order_items ── products
payments ───┘                  │
reviews ────┘                  └── sellers
```

Seven tables covering ~100k orders from 2016–2018 across Brazilian e-commerce marketplaces.
