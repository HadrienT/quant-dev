# Define variables for Terraform
TF_PLAN=tf.plan
TF_VARS=quantdev.tfvars
SECRETS_VARS=secrets.tfvars
TF_WORKSPACE=gcp-quant-dev

.PHONY: ensure-workspace
ensure-workspace:
	@terraform workspace select $(TF_WORKSPACE) >/dev/null 2>&1 || terraform workspace new $(TF_WORKSPACE) >/dev/null

# Default target
.PHONY: plan
plan: ensure-workspace
	terraform plan --out=$(TF_PLAN) --var-file=$(TF_VARS) --var-file=$(SECRETS_VARS)

# Optional: Add a target to apply the plan
.PHONY: apply
apply: ensure-workspace
	terraform apply $(TF_PLAN)

# Optional: Add a target to destroy resources
.PHONY: destroy
destroy:
	terraform destroy --var-file=$(TF_VARS) --var-file=$(SECRETS_VARS)

# Optional: Add a target to initialize Terraform
.PHONY: init
init:
	terraform init
