# resource "google_cloudfunctions_function" "auto_shutdown" {
#   name        = "auto-shutdown-vm"
#   runtime     = "python311"
#   region      = var.region
#   entry_point = "main"

#   source_archive_bucket = google_storage_bucket.auto_shutdown_function_code_bucket.name
#   source_archive_object = google_storage_bucket_object.auto_shutdown_function_code.name

#   available_memory_mb = 128
#   timeout             = 60
#   trigger_http        = true

#   environment_variables = {
#     PROJECT_ID = var.project_id
#     ZONE       = var.zone
#     VM_NAME    = var.vm_name
#   }
# }
