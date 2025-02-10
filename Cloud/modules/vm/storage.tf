#############################
# Autoshutdown function (does not work)
#############################
# resource "google_storage_bucket" "auto_shutdown_function_code_bucket" {
#   name          = "auto-shutdown-function-code-${var.project_id}"
#   location      = var.region
#   force_destroy = true
# }

# resource "google_storage_bucket_object" "auto_shutdown_function_code" {
#   name   = "function.zip"
#   bucket = google_storage_bucket.auto_shutdown_function_code_bucket.name
#   source = "${path.module}/../../scripts/autoshutdown.zip"
# }
