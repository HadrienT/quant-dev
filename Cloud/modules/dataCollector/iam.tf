#############################
# IAM: Allow public invocation of the function (gen2)
#############################
resource "google_cloudfunctions2_function_iam_member" "invoker" {
  project        = var.project_id
  location       = var.region
  cloud_function = google_cloudfunctions2_function.add_daily_function.name

  role   = "roles/run.invoker"
  member = "allUsers"
}
