"""
Generate an interactive HTML visualization of the BN with node tooltips showing CPT snippets.

Usage:
    python view_bn_html.py --cpts phase7_cpts.json --output bn_viz.html --network phase2_network_opt.json

Notes:
    - Requires pyvis (`pip install pyvis`).
    - Nodes are placed on a timeline (x = scheduled departure, y grouped by tail).
"""
import argparse
import json
import html
from pathlib import Path

try:
    from pyvis.network import Network
except ImportError:
    Network = None


def build_graph(cpts_path: str, output: str, network_path: str):
    if Network is None:
        raise ImportError("pyvis is required. Install with `pip install pyvis`.")
    data = json.loads(Path(cpts_path).read_text())
    cpts_raw = data["cpts"]
    res_cpds = data["resource_cpds"]
    num_bins = len(data["metadata"]["delay_bins"]) + 1

    flight_info = {}
    if network_path:
        net_data = json.loads(Path(network_path).read_text())
        for f in net_data["flights"]:
            flight_info[f["flight_id"]] = {
                "sd": f.get("scheduled_departure", 0),
                "tail": f.get("tail_id") or "TAIL_UNKNOWN",
            }

    # assign y per tail
    tail_to_y = {}
    for fid in res_cpds.keys():
        tail = flight_info.get(fid, {}).get("tail", "TAIL_UNKNOWN")
        if tail not in tail_to_y:
            tail_to_y[tail] = len(tail_to_y) * 50

    net = Network(height="1000px", width="100%", directed=True, notebook=False)
    net.set_options('{"physics": {"enabled": false}}')

    def node_pos(fid):
        info = flight_info.get(fid, {})
        x = info.get("sd", 0)
        y = tail_to_y.get(info.get("tail", "TAIL_UNKNOWN"), 0)
        return x, y

    def add_node(name, label, title, color, x=None, y=None):
        if name not in net.node_map:
            net.add_node(name, label=label, title=title, color=color, shape="dot", size=14, x=x, y=y, physics=False)

    def table_html(headers, rows):
        head = "".join(f"<th>{h}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
        )
        return f"<table border='1' cellpadding='2' cellspacing='0'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def add_open_link(table_str, title="Open full view"):
        safe = table_str.replace("`", "\\`")
        js = (
            "javascript:(function(){"
            "var w=window.open('','_blank','width=900,height=700,resizable=yes,scrollbars=yes');"
            f"w.document.write(`<html><body>{safe}</body></html>`);"
            "w.document.close();})();"
        )
        return f"<br><a href=\"{js}\">{title}</a>"

    # Build nodes and edges
    for fid, res in res_cpds.items():
        x, y = node_pos(fid)
        t_node = f"{fid}:t_bin"
        add_node(t_node, t_node, f"Departure delay bin for {fid}", "#ffcc00", x=x, y=y)
        add_node(f"{fid}:g", f"{fid}:g", "Ground cause category", "#cccccc", x=x - 10, y=y - 20)
        net.add_edge(f"{fid}:g", t_node)
        add_node(f"{fid}:k_bin", f"{fid}:k_bin", "Aircraft delay bin", "#99ccff", x=x - 10, y=y - 40)
        net.add_edge(f"{fid}:k_bin", t_node)
        add_node(f"{fid}:q_bin", f"{fid}:q_bin", "Pilot delay bin", "#99ff99", x=x - 30, y=y - 20)
        net.add_edge(f"{fid}:q_bin", t_node)
        add_node(f"{fid}:c_bin", f"{fid}:c_bin", "Cabin delay bin", "#ff9999", x=x - 30, y=y - 40)
        net.add_edge(f"{fid}:c_bin", t_node)
        for pfid in res.get("pax_parents", []):
            pnode = f"{fid}:p_{pfid}_bin"
            add_node(pnode, pnode, f"Pax delay bin from {pfid}", "#ffccff", x=x - 50, y=y - 20)
            net.add_edge(pnode, t_node)
        # upstream links
        for inbound in res.get("k", {}):
            net.add_edge(f"{inbound}:t_bin", f"{fid}:k_bin", color="#99ccff")
        for inbound in res.get("q", {}):
            net.add_edge(f"{inbound}:t_bin", f"{fid}:q_bin", color="#99ff99")
        for inbound in res.get("c", {}):
            net.add_edge(f"{inbound}:t_bin", f"{fid}:c_bin", color="#ff9999")
        for inbound in res.get("pax", {}):
            net.add_edge(f"{inbound}:t_bin", f"{fid}:p_{inbound}_bin", color="#ffccff")

        # CPT snippet for t_node
        table = cpts_raw.get(fid, {})
        parents_order = res.get("parents_order", [])
        header = ["parents"] + [f"t={i}" for i in range(num_bins)]
        rows = []
        for k, probs in list(table.items())[:8]:
            row = [k] + [round(probs.get(str(i), 0.0), 3) for i in range(num_bins)]
            rows.append(row)
        snippet_tbl = table_html(header, rows) if rows else "No rows"
        title = f"{t_node}<br>Parents: {parents_order}<br>{snippet_tbl}{add_open_link(snippet_tbl)}"
        if t_node in net.node_map:
            net.node_map[t_node]["title"] = title
        # g node info (data-driven prior if available)
        g_states = data.get("metadata", {}).get("g_states")
        g_priors = data.get("g_priors", {})
        if not g_states:
            sample_table = next(iter(cpts_raw.values())) if cpts_raw else {}
            g_states = sorted({k.split("|")[-1] for k in sample_table.keys()}) if sample_table else []
        if g_states and f"{fid}:g" in net.node_map:
            pri = g_priors.get(fid, {})
            rows_g = [[html.escape(s), round(pri.get(s, 0.0) if pri else 1.0 / len(g_states), 4)] for s in g_states]
            g_tbl = table_html(["g state", "prob"], rows_g)
            net.node_map[f"{fid}:g"]["title"] = f"{fid}:g<br>Ground cause prior<br>{g_tbl}{add_open_link(g_tbl, title='Open g table')}"

        # Probabilistic mapping snippets for resource nodes
        def mapping_table(mapping):
            header_m = ["t_in_bin"] + [f"bin {i}" for i in range(num_bins)]
            rows_m = []
            for tb, dist in list(mapping.items())[:8]:
                if isinstance(dist, dict):
                    dist_f = dist
                else:
                    # deterministic scalar -> one-hot
                    try:
                        b = int(dist)
                        dist_f = {b: 1.0}
                    except Exception:
                        dist_f = {}
                row_m = [tb] + [round(dist_f.get(str(i), dist_f.get(i, 0.0)), 4) for i in range(num_bins)]
                rows_m.append(row_m)
            if not rows_m:
                rows_m.append(["-"] + ["-"] * num_bins)
            tbl = table_html(header_m, rows_m)
            return f"{tbl}{add_open_link(tbl, title='Open mapping')}"

        if res.get("k"):
            for inbound, mapping in res.get("k", {}).items():
                node_name = f"{fid}:k_bin"
                if node_name in net.node_map:
                    net.node_map[node_name]["title"] = f"{node_name}<br>Mapping from {inbound}:t_bin<br>{mapping_table(mapping)}"
        if res.get("q"):
            for inbound, mapping in res.get("q", {}).items():
                node_name = f"{fid}:q_bin"
                if node_name in net.node_map:
                    net.node_map[node_name]["title"] = f"{node_name}<br>Mapping from {inbound}:t_bin<br>{mapping_table(mapping)}"
        if res.get("c"):
            for inbound, mapping in res.get("c", {}).items():
                node_name = f"{fid}:c_bin"
                if node_name in net.node_map:
                    net.node_map[node_name]["title"] = f"{node_name}<br>Mapping from {inbound}:t_bin<br>{mapping_table(mapping)}"
        if res.get("pax"):
            for inbound, mapping in res.get("pax", {}).items():
                node_name = f"{fid}:p_{inbound}_bin"
                if node_name in net.node_map:
                    net.node_map[node_name]["title"] = f"{node_name}<br>Mapping from {inbound}:t_bin<br>{mapping_table(mapping)}"

    net.write_html(output, open_browser=False, notebook=False)
    print(f"Wrote HTML to {output}")


def main():
    parser = argparse.ArgumentParser(description="Visualize BN as HTML with pyvis.")
    parser.add_argument("--cpts", default="phase7_cpts.json", help="CPT JSON file")
    parser.add_argument("--network", default=None, help="Optional network JSON to position nodes by time/tail")
    parser.add_argument("--output", default="bn_viz.html", help="Output HTML file")
    args = parser.parse_args()
    build_graph(args.cpts, args.output, args.network)


if __name__ == "__main__":
    main()
