"""
Graph construction from PS4 event log.

Explicit graph:  user → resource  (directed bipartite, edges = access events)
Implicit graph:  user → user      (connected via shared resource access = peer clusters)

Node embeddings are 8-dimensional structural feature vectors per user.
"""

import numpy as np
import pandas as pd
import networkx as nx


SENSITIVITY_SCORE = {"low": 1, "medium": 2, "high": 3}
PRIVILEGE_SCORE = {"user": 0, "power-user": 1, "admin": 2, "service-account": 3}


class GraphBuilder:
    def __init__(self):
        self.explicit: nx.DiGraph = None   # user → resource
        self.implicit: nx.Graph = None     # user ↔ user via shared resources
        self._embeddings: dict[str, np.ndarray] = {}

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, logs: pd.DataFrame, profiles: pd.DataFrame) -> "GraphBuilder":
        logs = logs.copy()
        logs["timestamp"] = pd.to_datetime(logs["timestamp"])

        self.explicit = self._build_explicit(logs, profiles)
        self.implicit = self._build_implicit(logs, profiles)
        self._embeddings = self._compute_embeddings(logs, profiles)
        return self

    def _build_explicit(self, logs: pd.DataFrame, profiles: pd.DataFrame) -> nx.DiGraph:
        G = nx.DiGraph()

        # User nodes
        for _, p in profiles.iterrows():
            G.add_node(
                p["user_id"],
                node_type="user",
                department=p.get("department", ""),
                privilege_level=p.get("privilege_level", "user"),
                days_inactive=int(p.get("days_inactive", 0)),
            )

        # Resource nodes: sensitivity from mode of all events touching that resource
        for res in logs["resource"].unique():
            mask = logs["resource"] == res
            sens = logs.loc[mask, "resource_sensitivity"].mode()
            sens_val = sens.iloc[0] if len(sens) > 0 else "low"
            G.add_node(
                res,
                node_type="resource",
                sensitivity_label=sens_val,
                sensitivity=SENSITIVITY_SCORE.get(sens_val, 1),
            )

        # Edges: user → resource, weight = number of accesses
        edge_agg = (
            logs.groupby(["user_id", "resource"])
            .agg(
                weight=("action", "count"),
                export_count=("action", lambda x: (x == "export_data").sum()),
                failure_count=("status", lambda x: (x == "failure").sum()),
                last_access=("timestamp", "max"),
            )
            .reset_index()
        )
        for _, row in edge_agg.iterrows():
            if G.has_node(row["user_id"]) and G.has_node(row["resource"]):
                G.add_edge(
                    row["user_id"], row["resource"],
                    weight=int(row["weight"]),
                    export_count=int(row["export_count"]),
                    failure_count=int(row["failure_count"]),
                    last_access=str(row["last_access"]),
                )

        return G

    def _build_implicit(self, logs: pd.DataFrame, profiles: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()

        for _, p in profiles.iterrows():
            G.add_node(
                p["user_id"],
                department=p.get("department", ""),
                privilege_level=p.get("privilege_level", "user"),
            )

        # Connect users who share at least one resource
        resource_users = logs.groupby("resource")["user_id"].apply(set).to_dict()
        for resource, users in resource_users.items():
            users = list(users)
            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    u, v = users[i], users[j]
                    if not (G.has_node(u) and G.has_node(v)):
                        continue
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += 1
                        G[u][v]["shared_resources"].append(resource)
                    else:
                        G.add_edge(u, v, weight=1, shared_resources=[resource])

        return G

    def _compute_embeddings(
        self, logs: pd.DataFrame, profiles: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """
        8-dim structural embedding per user:
          [out_degree, total_access_weight, high_sens_accesses,
           peer_degree, peer_weight, privilege_score, days_inactive, unique_ips]
        """
        embeddings = {}
        G = self.explicit
        impl = self.implicit

        for _, p in profiles.iterrows():
            uid = p["user_id"]

            out_degree = G.out_degree(uid) if G.has_node(uid) else 0

            total_weight = (
                sum(d.get("weight", 0) for _, _, d in G.out_edges(uid, data=True))
                if G.has_node(uid) else 0
            )

            high_sens = (
                sum(
                    1 for _, res, _ in G.out_edges(uid, data=True)
                    if G.has_node(res) and G.nodes[res].get("sensitivity", 0) >= 3
                )
                if G.has_node(uid) else 0
            )

            peer_degree = impl.degree(uid) if impl.has_node(uid) else 0
            peer_weight = (
                sum(d.get("weight", 0) for _, _, d in impl.edges(uid, data=True))
                if impl.has_node(uid) else 0
            )

            priv_score = PRIVILEGE_SCORE.get(str(p.get("privilege_level", "user")), 0)
            days_inactive = min(int(p.get("days_inactive", 0)), 365)
            unique_ips = int(logs[logs["user_id"] == uid]["source_ip"].nunique())

            embeddings[uid] = np.array(
                [out_degree, total_weight, high_sens, peer_degree,
                 peer_weight, priv_score, days_inactive, unique_ips],
                dtype=np.float32,
            )

        return embeddings

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_embedding(self, user_id: str) -> np.ndarray:
        return self._embeddings.get(user_id, np.zeros(8, dtype=np.float32))

    def get_graph_divergence(self, user_id: str, department: str,
                              profiles: pd.DataFrame) -> float:
        """
        Mahalanobis-style distance of user embedding from their department peer centroid.
        Normalised so ~1.0 = typical, > 2.0 = notably divergent.
        """
        emb = self.get_embedding(user_id)
        peer_ids = profiles[
            (profiles["department"] == department) &
            (profiles["user_id"] != user_id)
        ]["user_id"].tolist()

        peer_embs = np.array(
            [self.get_embedding(pid) for pid in peer_ids if pid in self._embeddings]
        )
        if len(peer_embs) < 2:
            return 0.0

        centroid = peer_embs.mean(axis=0)
        spread = peer_embs.std(axis=0)
        spread = np.where(spread < 1e-6, 1.0, spread)
        return float(np.linalg.norm((emb - centroid) / spread))

    def get_user_resources(self, user_id: str) -> list[str]:
        """List of resources the user has accessed, sorted by access count."""
        if not self.explicit or not self.explicit.has_node(user_id):
            return []
        edges = sorted(
            self.explicit.out_edges(user_id, data=True),
            key=lambda x: x[2].get("weight", 0),
            reverse=True,
        )
        return [res for _, res, _ in edges]

    def subgraph_for_display(self, user_id: str, depth: int = 1) -> nx.DiGraph:
        """Return a small ego subgraph centred on user_id for visualisation."""
        if not self.explicit or not self.explicit.has_node(user_id):
            return nx.DiGraph()
        nodes = {user_id}
        for _, res in self.explicit.out_edges(user_id):
            nodes.add(res)
            if depth > 1:
                for other_user in self.explicit.predecessors(res):
                    nodes.add(other_user)
        return self.explicit.subgraph(nodes).copy()
