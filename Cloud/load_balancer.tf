
resource "google_compute_region_network_endpoint_group" "cloud_run_neg" {
  name                  = "cloud-run-neg"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  cloud_run {
    service = google_cloud_run_v2_service.service.name
  }
}

resource "google_compute_global_address" "lb_ip" {
  name = "cloud-run-lb-ip"
}


resource "google_compute_url_map" "cloud_run_url_map" {
  name            = "cloud-run-url-map"
  default_service = google_compute_backend_service.cloud_run_backend.id

  host_rule {
    hosts        = [var.domain_name]
    path_matcher = "cloud-run-matcher"
  }

  path_matcher {
    name            = "cloud-run-matcher"
    default_service = google_compute_backend_service.cloud_run_backend.id
  }
}

resource "google_compute_url_map" "redirect_http_to_https" {
  name = "redirect-http-to-https"

  default_url_redirect {
    strip_query            = false
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    https_redirect         = true
  }
}


resource "google_compute_target_http_proxy" "http_proxy" {
  name    = "cloud-run-http-proxy"
  url_map = google_compute_url_map.redirect_http_to_https.id
}


resource "google_compute_managed_ssl_certificate" "ssl_certificate" {
  name    = "ssl-${replace(var.domain_name, ".", "-")}"
  project = var.project_id

  managed {
    domains = [var.domain_name]
  }
}


resource "google_compute_target_https_proxy" "https_proxy" {
  name             = "cloud-run-https-proxy"
  url_map          = google_compute_url_map.cloud_run_url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.ssl_certificate.id]
}

resource "google_compute_global_forwarding_rule" "http_forwarding_rule" {
  name                  = "http-forwarding-rule"
  load_balancing_scheme = "EXTERNAL"
  target                = google_compute_target_http_proxy.http_proxy.id
  port_range            = "80"
  ip_address            = google_compute_global_address.lb_ip.address
}

resource "google_compute_global_forwarding_rule" "https_forwarding_rule" {
  name                  = "https-forwarding-rule"
  load_balancing_scheme = "EXTERNAL"
  target                = google_compute_target_https_proxy.https_proxy.id
  port_range            = "443"
  ip_address            = google_compute_global_address.lb_ip.address
}


resource "google_compute_firewall" "allow_lb_health_checks" {
  name    = "allow-lb-health-checks"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"] # IPs des Load Balancers Google
}

resource "google_compute_security_policy" "block_googlebot" {
  name = "block-googlebot-policy"

  # Rule to block Googlebot
  rule {
    action   = "deny(403)"
    priority = 1000
    match {
      expr {
        expression = "has(request.headers['user-agent']) && request.headers['user-agent'].contains('Googlebot')"
      }
    }
    description = "Block Googlebot user agent"
  }

  # Default rule (required)
  rule {
    action   = "allow"
    priority = 2147483647
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default rule, higher priority overrides it"
  }
}

resource "google_compute_backend_service" "cloud_run_backend" {
  name                  = "cloud-run-backend"
  load_balancing_scheme = "EXTERNAL"
  protocol              = "HTTPS"
  timeout_sec           = 30
  enable_cdn            = false

  backend {
    group = google_compute_region_network_endpoint_group.cloud_run_neg.id
  }

  security_policy = google_compute_security_policy.block_googlebot.id

  port_name = "http" # Required for Cloud Run

  custom_request_headers = [
    "Host: ${google_cloud_run_v2_service.service.name}-${google_cloud_run_v2_service.service.project}.a.run.app"
  ]

  custom_response_headers = [
    "Access-Control-Allow-Origin: *",
    "Access-Control-Allow-Methods: GET, POST, OPTIONS",
    "Access-Control-Allow-Headers: DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range",
    "Access-Control-Expose-Headers: Content-Length,Content-Range"
  ]

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}
