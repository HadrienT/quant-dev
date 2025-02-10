#############################
# BigQuery datasets and tables to store the data
#############################
resource "google_bigquery_dataset" "dataset" {
  dataset_id = "financial_data"
  location   = "EU"
}

resource "google_bigquery_table" "table" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "sp500_data"
  project    = google_bigquery_dataset.dataset.project

  schema = jsonencode([
    { name = "Date", type = "DATE", mode = "REQUIRED" },
    { name = "Ticker", type = "STRING", mode = "REQUIRED" },
    { name = "Open", type = "FLOAT", mode = "NULLABLE" },
    { name = "High", type = "FLOAT", mode = "NULLABLE" },
    { name = "Low", type = "FLOAT", mode = "NULLABLE" },
    { name = "Close", type = "FLOAT", mode = "NULLABLE" },
    { name = "Volume", type = "INTEGER", mode = "NULLABLE" }
  ])
}

resource "google_bigquery_dataset" "dataset_test" {
  dataset_id = "financial_data_test"
  location   = "EU"
}

resource "google_bigquery_table" "table_test" {
  dataset_id = google_bigquery_dataset.dataset_test.dataset_id
  table_id   = "sp500_data_test"
  project    = google_bigquery_dataset.dataset_test.project

  schema = jsonencode([
    { name = "Date", type = "DATE", mode = "REQUIRED" },
    { name = "Ticker", type = "STRING", mode = "REQUIRED" },
    { name = "Open", type = "FLOAT", mode = "NULLABLE" },
    { name = "High", type = "FLOAT", mode = "NULLABLE" },
    { name = "Low", type = "FLOAT", mode = "NULLABLE" },
    { name = "Close", type = "FLOAT", mode = "NULLABLE" },
    { name = "Volume", type = "INTEGER", mode = "NULLABLE" }
  ])
}
