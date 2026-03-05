resource "google_cloud_run_v2_service" "api" {
  name                = var.service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}/${var.service_name}:latest"

      env {
        name  = "BQ_PROJECT_ID"
        value = var.bq_project_id
      }
      env {
        name  = "BQ_DATASET_ID"
        value = var.bq_dataset_id
      }
      env {
        name  = "BQ_TABLE_ID"
        value = var.bq_table_id
      }
      env {
        name  = "FRED_API_KEY"
        value = var.fred_api_key
      }
      env {
        name  = "API_KEY"
        value = var.api_key
      }

      resources {
        limits = {
          memory = "2Gi"
          cpu    = "2"
        }
      }

      ports {
        container_port = 8000
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 2
    }

    timeout = "600s"
  }
}

# Allow only the web service (and Cloud Build) to invoke the API.
# Not allUsers — the web service proxies requests through nginx.
resource "google_cloud_run_service_iam_binding" "api_invoker" {
  location = google_cloud_run_v2_service.api.location
  project  = google_cloud_run_v2_service.api.project
  service  = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}
