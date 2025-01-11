resource "google_compute_address" "static_ip" {
  name   = "ht-vm-e2-medium-static-ip"
  region = "europe-west3"
}

resource "google_compute_instance" "instance-20250109-093047" {
  name         = "ht-vm-e2-medium"
  machine_type = "e2-medium"
  zone         = "europe-west3-a"

  boot_disk {
    auto_delete = true
    device_name = "ht-vm-e2-medium-dsk"

    initialize_params {
      image = "projects/debian-cloud/global/images/debian-12-bookworm-v20241210"
      size  = 30
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  labels = {
    goog-ec-src = "vm_add-tf"
  }

  network_interface {
    subnetwork = "projects/quant-dev-442615/regions/europe-west3/subnetworks/default"

    access_config {
      network_tier = "PREMIUM"
      nat_ip       = google_compute_address.static_ip.address
    }
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
  }

  service_account {
    email  = "956975863647-compute@developer.gserviceaccount.com"
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append"
    ]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  metadata = {
    ssh-keys = "ht-vm-e2-medium:${var.ssh_key}"
  }
}
