output "cloud_run_service_url" {
  description = "URL of the Cloud Run Service"
  value       = google_cloud_run_v2_service.service.uri
}
