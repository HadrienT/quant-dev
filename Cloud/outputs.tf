output "function_url" {
  description = "URL de la Cloud Function"
  value       = google_cloudfunctions_function.add_daily_function.https_trigger_url
}

output "scheduler_job_name" {
  description = "Nom du job Cloud Scheduler"
  value       = google_cloud_scheduler_job.daily_sp500_job.name
}


output "github_owner" {
  value = var.github_owner
}

output "github_repo_name" {
  value = var.github_repo_name
}
