#############################
# Cloud Run Service
#############################
resource "google_cloud_run_v2_service" "service" {
  name                = var.service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_name}/${var.service_name}:latest"
      env {
        name  = "FRED_API_KEY"
        value = var.fred_api_key
      }
      resources {
        limits = {
          memory = "1024Mi"
          cpu    = "1"
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    timeout = "600s"
  }
}
