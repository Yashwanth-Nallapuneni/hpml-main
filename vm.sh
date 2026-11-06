#!/usr/bin/env bash
# Team 20 VM helper. Finds the team VM by label, so it keeps working if the VM is replaced.
# Usage: bash vm.sh [ssh|start|stop|status]   (default: ssh)
set -euo pipefail
PROJECT=hpml-team20
read -r NAME ZONE STATUS < <(gcloud compute instances list --project=$PROJECT \
  --filter="labels.team=team20 AND labels.role=gpu" --format="value(name,zone.basename(),status)" | head -1)
if [ -z "${NAME:-}" ]; then echo "No team VM found in $PROJECT (label team=team20)"; exit 1; fi
case "${1:-ssh}" in
  status) echo "$NAME  $ZONE  $STATUS" ;;
  start)  gcloud compute instances start "$NAME" --zone="$ZONE" --project=$PROJECT ;;
  stop)   gcloud compute instances stop  "$NAME" --zone="$ZONE" --project=$PROJECT ;;
  ssh)    [ "$STATUS" = RUNNING ] || { echo "$NAME is $STATUS. Run: bash vm.sh start"; exit 1; }
          gcloud compute ssh "$NAME" --zone="$ZONE" --project=$PROJECT --tunnel-through-iap ;;
  *)      echo "usage: bash vm.sh [ssh|start|stop|status]"; exit 1 ;;
esac
