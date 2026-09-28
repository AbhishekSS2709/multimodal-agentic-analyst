#!/usr/bin/env bash
# One-time Google Cloud setup so GitHub Actions can deploy to Cloud Run
# WITHOUT a stored service-account key (Workload Identity Federation).
#
# Run in Cloud Shell, in the project that hosts the service:
#   bash deploy/gcp_github_deploy_setup.sh
#
# What it creates:
#   * service account  github-deployer  -- allowed to deploy Cloud Run, nothing else
#   * workload identity pool "github" + OIDC provider "github-oidc" that trusts
#     GitHub's token issuer, restricted to ONE repo and ONE branch (main)
# Re-running is safe: existing resources are left as they are.
set -euo pipefail

REPO="${REPO:-AbhishekSS2709/multimodal-agentic-analyst}"
PROJECT="$(gcloud config get-value project 2>/dev/null)"
PN="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
SA_NAME=github-deployer
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
RUNTIME_SA="$PN-compute@developer.gserviceaccount.com"
POOL=github
PROVIDER=github-oidc

echo "Project: $PROJECT ($PN)   Repo: $REPO"

gcloud services enable iamcredentials.googleapis.com sts.googleapis.com iam.googleapis.com --quiet

gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name="GitHub Actions deployer" --quiet

# Least privilege: deploy Cloud Run revisions, read the image mirror, and run
# the service as its existing runtime identity -- no project-wide admin.
for ROLE in roles/run.developer roles/artifactregistry.reader; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" \
    --role="$ROLE" --condition=None --quiet --format=none
done
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member="serviceAccount:$SA" --role=roles/iam.serviceAccountUser --quiet --format=none

gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools create "$POOL" --location=global --display-name="GitHub Actions" --quiet

gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" >/dev/null 2>&1 || \
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --location=global --workload-identity-pool="$POOL" \
    --display-name="GitHub OIDC" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition="assertion.repository=='$REPO' && assertion.ref=='refs/heads/main'" \
    --quiet

gcloud iam service-accounts add-iam-policy-binding "$SA" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PN/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPO" \
  --quiet --format=none

echo
echo "Done. Values used by .github/workflows/ci.yml:"
echo "  WIF provider:    projects/$PN/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"
echo "  Service account: $SA"
echo "Now set the GitHub repository variable CLOUD_RUN_DEPLOY=true to turn deploys on."
