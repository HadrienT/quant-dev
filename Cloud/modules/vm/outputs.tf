output "vm_instance_id" {
  description = "ID of the VM instance"
  value       = google_compute_instance.instance.id
}

# output "auto_shutdown_function_url" {
#   description = "URL of the auto-shutdown Cloud Function"
#   value       = google_cloudfunctions_function.auto_shutdown.https_trigger_url
# }
