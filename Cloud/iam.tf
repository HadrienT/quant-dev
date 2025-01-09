resource "google_cloudfunctions_function_iam_member" "invoker" {
  project = var.project_id
  region  = var.region
  cloud_function = google_cloudfunctions_function.add_daily_function.name

  role   = "roles/cloudfunctions.invoker"
  member = "allUsers" 
}
