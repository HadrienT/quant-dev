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
        name  = "API_URL"
        value = var.api_url
      }
      resources {
        limits = {
          memory = "512Mi"
          cpu    = "1"
        }
      }

      ports {
        container_port = 8080
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }

    timeout = "60s"
  }
}
