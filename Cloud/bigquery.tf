resource "google_bigquery_dataset" "dataset" {
  dataset_id = "financial_data" # Nom du dataset
  location   = "EU"             # Localisation du dataset (Europe)
}

resource "google_bigquery_table" "table" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id # Associe la table au dataset
  table_id   = "sp500_data"                               # Nom de la table
  project    = google_bigquery_dataset.dataset.project    # Projet contenant le dataset

  schema = jsonencode([
    {
      name = "Date"
      type = "DATE"
      mode = "REQUIRED"
    },
    {
      name = "Ticker"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "Open"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "High"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "Low"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "Close"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "Volume"
      type = "INTEGER"
      mode = "NULLABLE"
    }
  ])
}
