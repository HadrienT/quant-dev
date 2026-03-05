resource "google_service_account" "bq_reader" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = var.service_account_display_name
}

resource "google_project_iam_member" "bq_data_viewer" {
  project = var.project_id
  role    = "roles/bigquery.dataViewer"
  member  = "serviceAccount:${google_service_account.bq_reader.email}"
}

resource "google_project_iam_member" "bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.bq_reader.email}"
}

resource "google_storage_bucket" "portfolio_bucket" {
  project                     = var.project_id
  name                        = var.portfolio_bucket_name
  location                    = var.portfolio_bucket_location
  uniform_bucket_level_access = true
  force_destroy               = true

  lifecycle_rule {
    condition { age = 365 }
    action { type = "Delete" }
  }
}

resource "google_storage_bucket_iam_member" "portfolio_bucket_admin" {
  bucket = google_storage_bucket.portfolio_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.bq_reader.email}"
}

resource "google_service_account_key" "bq_reader_key" {
  service_account_id = google_service_account.bq_reader.name
}

resource "local_file" "bq_reader_key_file" {
  filename = var.key_output_path
  content  = base64decode(google_service_account_key.bq_reader_key.private_key)
}
