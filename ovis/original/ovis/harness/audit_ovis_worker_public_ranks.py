"""Export each CFG rank's real audit instead of relying on the primary reply."""

import json
from pathlib import Path

from harness.audit_ovis_worker_public_latency import OvisPublicLatencyAudit


class OvisPublicRankAudit(OvisPublicLatencyAudit):
    def _export_rank(self, result, path):
        assert result["rank"] == self.rank
        destination = Path(f"{path}-rank{self.rank}.json")
        with destination.open("x") as output:
            json.dump(result, output, indent=2)
        return result

    def pipeline_snapshot(self, path):
        return self._export_rank(super().pipeline_snapshot(), path)

    def ovis_end_audit(self, path):
        return self._export_rank(super().ovis_end_audit(), path)

    def ovis_exception_cleanup(self, path):
        return self._export_rank(super().ovis_exception_cleanup(), path)
