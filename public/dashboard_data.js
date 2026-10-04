window.DASHBOARD_DATA = {
  "generated_at": "2026-10-04 12:09:11",
  "dataset_overview": {
    "rows": 7043,
    "columns": 24,
    "total_missing": 0,
    "total_cells": 169032,
    "missing_pct": 0.0,
    "columns_info": [
      {
        "name": "customer_id",
        "dtype": "str",
        "missing": 0,
        "unique": 7043
      },
      {
        "name": "gender",
        "dtype": "str",
        "missing": 0,
        "unique": 2
      },
      {
        "name": "seniorcitizen",
        "dtype": "int64",
        "missing": 0,
        "unique": 2
      },
      {
        "name": "partner",
        "dtype": "str",
        "missing": 0,
        "unique": 2
      },
      {
        "name": "dependents",
        "dtype": "str",
        "missing": 0,
        "unique": 2
      },
      {
        "name": "tenure",
        "dtype": "int64",
        "missing": 0,
        "unique": 73
      },
      {
        "name": "phoneservice",
        "dtype": "str",
        "missing": 0,
        "unique": 2
      },
      {
        "name": "multiplelines",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "internetservice",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "onlinesecurity",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "onlinebackup",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "deviceprotection",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "techsupport",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "streamingtv",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "streamingmovies",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "contract",
        "dtype": "str",
        "missing": 0,
        "unique": 3
      },
      {
        "name": "paperlessbilling",
        "dtype": "str",
        "missing": 0,
        "unique": 2
      },
      {
        "name": "paymentmethod",
        "dtype": "str",
        "missing": 0,
        "unique": 4
      },
      {
        "name": "monthlycharges",
        "dtype": "float64",
        "missing": 0,
        "unique": 1585
      },
      {
        "name": "totalcharges",
        "dtype": "float64",
        "missing": 0,
        "unique": 6531
      },
      {
        "name": "churn",
        "dtype": "str",
        "missing": 0,
        "unique": 2
      },
      {
        "name": "tenure_group",
        "dtype": "str",
        "missing": 0,
        "unique": 5
      },
      {
        "name": "avg_monthly_spend",
        "dtype": "float64",
        "missing": 0,
        "unique": 6639
      },
      {
        "name": "revenue",
        "dtype": "float64",
        "missing": 0,
        "unique": 1585
      }
    ],
    "memory_mb": 6.97,
    "duplicate_rows": 0
  },
  "churn_distribution": {
    "churned": 2887,
    "retained": 4156,
    "total": 7043,
    "churn_rate": 40.99,
    "retention_rate": 59.01
  },
  "revenue_analysis": {
    "total_monthly_charges": 0.0,
    "avg_monthly_charges": 0.0,
    "total_revenue": 0.0,
    "avg_revenue_per_customer": 0.0,
    "churned_total_revenue": 0.0,
    "retained_total_revenue": 0.0,
    "charge_distribution": {
      "$0-30": 0,
      "$30-50": 0,
      "$50-70": 0,
      "$70-90": 0,
      "$90-120": 0
    }
  },
  "segment_analysis": {
    "contract": [
      {
        "contract": "month-to-month",
        "count": 3875,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      },
      {
        "contract": "one year",
        "count": 1473,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      },
      {
        "contract": "two year",
        "count": 1695,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      }
    ],
    "gender": [],
    "senior_citizen": []
  },
  "tenure_analysis": {
    "groups": [
      {
        "group": "0-6 mo",
        "count": 1470,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      },
      {
        "group": "6-12 mo",
        "count": 705,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      },
      {
        "group": "12-24 mo",
        "count": 1024,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      },
      {
        "group": "24-48 mo",
        "count": 1594,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      },
      {
        "group": "48+ mo",
        "count": 2239,
        "churn_rate": 0.0,
        "avg_monthly": 0.0
      }
    ],
    "avg_tenure": 32.37114865824223,
    "median_tenure": 29.0,
    "max_tenure": 72.0
  },
  "services_analysis": {
    "internet_service": [],
    "payment_method": []
  },
  "model": {
    "evaluation": {
      "roc_auc": 0.8447,
      "precision": 0.5,
      "recall": 0.7,
      "f1_score": 0.6,
      "confusion_matrix": [
        [
          0,
          0
        ],
        [
          0,
          0
        ]
      ],
      "test_samples": 1409
    },
    "model_comparison": {
      "random_forest": {
        "roc_auc": 0.8447,
        "precision": 0.5,
        "recall": 0.7
      },
      "logistic_regression": {
        "roc_auc": 0.8,
        "precision": 0.5,
        "recall": 0.7
      }
    },
    "selected_model": "random_forest",
    "model_version": "v2.0",
    "num_features": 29,
    "feature_list": [
      "customer_id",
      "gender",
      "seniorcitizen",
      "partner",
      "dependents",
      "tenure",
      "phoneservice",
      "multiplelines",
      "internetservice",
      "onlinesecurity"
    ],
    "thresholds": [
      {
        "threshold": 0.3,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5
      },
      {
        "threshold": 0.4,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5
      },
      {
        "threshold": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5
      },
      {
        "threshold": 0.6,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5
      },
      {
        "threshold": 0.7,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5
      },
      {
        "threshold": 0.8,
        "precision": 0.5,
        "recall": 0.5,
        "f1_score": 0.5
      }
    ],
    "risk_distribution": {
      "LOW": 3558,
      "MEDIUM": 1748,
      "HIGH": 1737
    },
    "top_customers": [
      {
        "customer_id": "9090-sgqxl",
        "churn_probability": 0.6914,
        "revenue": 7299.65,
        "expected_loss": 5047.08,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "2452-kdrrh",
        "churn_probability": 0.706,
        "revenue": 6841.05,
        "expected_loss": 4829.52,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "3761-flyzi",
        "churn_probability": 0.6931,
        "revenue": 7082.45,
        "expected_loss": 4909.09,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "5647-fxotp",
        "churn_probability": 0.7245,
        "revenue": 6401.25,
        "expected_loss": 4637.56,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "2378-vtkdh",
        "churn_probability": 0.7076,
        "revenue": 6578.55,
        "expected_loss": 4655.02,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "7901-tbkjx",
        "churn_probability": 0.7655,
        "revenue": 5594.0,
        "expected_loss": 4282.47,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "1013-qcwam",
        "churn_probability": 0.6987,
        "revenue": 6690.75,
        "expected_loss": 4674.68,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "0946-cljti",
        "churn_probability": 0.7491,
        "revenue": 5812.6,
        "expected_loss": 4354.44,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "0324-brpcj",
        "churn_probability": 0.686,
        "revenue": 6851.65,
        "expected_loss": 4700.05,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "4550-vbofe",
        "churn_probability": 0.6733,
        "revenue": 7101.5,
        "expected_loss": 4781.55,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "7056-imhcc",
        "churn_probability": 0.7593,
        "revenue": 5549.4,
        "expected_loss": 4213.43,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "1035-ipqpu",
        "churn_probability": 0.6998,
        "revenue": 6479.4,
        "expected_loss": 4534.51,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "4553-dvpzg",
        "churn_probability": 0.7164,
        "revenue": 6164.7,
        "expected_loss": 4416.45,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "8634-mphtr",
        "churn_probability": 0.8038,
        "revenue": 4871.05,
        "expected_loss": 3915.38,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "4433-jcgcg",
        "churn_probability": 0.816,
        "revenue": 4680.05,
        "expected_loss": 3819.08,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "6173-golsu",
        "churn_probability": 0.7122,
        "revenue": 6079.0,
        "expected_loss": 4329.46,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "3791-lgqcy",
        "churn_probability": 0.7356,
        "revenue": 5688.05,
        "expected_loss": 4184.26,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "6646-vrfol",
        "churn_probability": 0.7481,
        "revenue": 5485.5,
        "expected_loss": 4103.68,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "6377-whaox",
        "churn_probability": 0.6916,
        "revenue": 6411.25,
        "expected_loss": 4434.26,
        "monthly_charges": 0.0,
        "tenure": 0
      },
      {
        "customer_id": "2675-dhutr",
        "churn_probability": 0.7264,
        "revenue": 5780.7,
        "expected_loss": 4199.28,
        "monthly_charges": 0.0,
        "tenure": 0
      }
    ],
    "business_kpis": {
      "total_customers": 7043,
      "high_risk_pct": 24.66,
      "total_revenue": 0.0,
      "revenue_at_risk": 5099675.26,
      "revenue_at_risk_pct": 0.0
    }
  }
};
