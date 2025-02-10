#############################
# Job Cloud Scheduler to trigger auto shutdown
#############################
# resource "google_cloud_scheduler_job" "auto_shutdown_job" {
#   name        = "auto-shutdown-scheduler"
#   schedule    = "*/5 * * * *"
#   description = "Check idle VM and trigger auto shutdown"

#   http_target {
#     uri         = google_cloudfunctions_function.auto_shutdown.https_trigger_url
#     http_method = "POST"
#   }
# }
