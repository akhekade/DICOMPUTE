const $ = (id) => document.getElementById(id);

function fmtMem(gb) {
  if (gb == null) return "—";
  return gb >= 10 ? `${Math.round(gb)} GB` : `${gb.toFixed(1)} GB`;
}

function renderNodes(nodes) {
  const root = $("node-list");
  if (!nodes.length) {
    root.innerHTML = `<p class="empty">No providers online yet. On another Mac Mini or laptop run:<br><code>dico join --coordinator ${location.origin} --transport websocket</code></p>`;
    return;
  }
  root.innerHTML = nodes
    .map((n, i) => {
      const caps = n.capabilities || {};
      const cap = n.capacity || {};
      const accel = [
        caps.has_metal ? "Metal" : null,
        caps.has_cuda ? "CUDA" : null,
        !caps.has_metal && !caps.has_cuda ? "CPU" : null,
      ]
        .filter(Boolean)
        .join(" · ");
      const role = n.role === "coordinator" ? "coordinator" : "provider";
      const transport = n.transport || "http";
      const slots =
        cap.max_slots != null
          ? `${cap.free_slots ?? "?"}/${cap.max_slots} slots`
          : "slots n/a";
      return `
        <article class="node" style="animation-delay:${i * 40}ms">
          <div class="node-id">
            ${n.node_id}
            <small>${caps.hostname || "host"} · ${caps.platform || "unknown"} · ${accel}</small>
          </div>
          <div class="badge ${role}">${role} · ${transport} · load ${(n.load ?? 0).toFixed(2)}</div>
          <div class="meta">${caps.cpu_cores || "?"} cores · ${slots} · model v${n.model_version ?? 0}</div>
        </article>
      `;
    })
    .join("");
}

async function refresh() {
  try {
    const res = await fetch("/status");
    if (!res.ok) throw new Error(`status ${res.status}`);
    const data = await res.json();
    $("m-providers").textContent = String(data.online_providers ?? 0);
    $("m-version").textContent = `v${data.model_version ?? 0}`;
    $("m-cores").textContent = String(data.total_cpu_cores ?? 0);
    $("m-mem").textContent = fmtMem(data.total_memory_gb);
    const wsEl = $("m-ws");
    if (wsEl) wsEl.textContent = String(data.ws_providers ?? 0);
    renderNodes(data.nodes || []);
    $("pulse").classList.add("live");
    $("pulse-label").textContent = data.auth_required ? "mesh live · auth" : "mesh live";

  } catch (err) {
    $("pulse").classList.remove("live");
    $("pulse-label").textContent = "unreachable";
    console.error(err);
  }
}

function showOut(obj) {
  const el = $("action-out");
  el.hidden = false;
  el.textContent = JSON.stringify(obj, null, 2);
}

async function trainRound() {
  const btn = $("btn-train");
  btn.disabled = true;
  try {
    const res = await fetch("/train/round", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 3, samples: 256, learning_rate: 0.05, batch_size: 32 }),
    });
    const data = await res.json();
    showOut({
      model_version: data.model_version,
      num_contributors: data.num_contributors,
      checksum_sha256: data.checksum_sha256,
    });

    await refresh();
  } catch (err) {
    showOut({ error: String(err) });
  } finally {
    btn.disabled = false;
  }
}

async function sampleInfer() {
  const btn = $("btn-infer");
  btn.disabled = true;
  try {
    const features = Array.from({ length: 8 }, () => Number((Math.random() * 2 - 1).toFixed(3)));
    const res = await fetch("/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features, ensemble: true }),
    });
    const data = await res.json();
    showOut({
      prediction: data.prediction,
      probabilities: data.probabilities,
      strategy: data.strategy,
      cost_micro_usd: data.cost_micro_usd,
      contributors: (data.contributors || []).map((c) => c.node_id),
    });

  } catch (err) {
    showOut({ error: String(err) });
  } finally {
    btn.disabled = false;
  }
}

$("btn-train").addEventListener("click", trainRound);
$("btn-infer").addEventListener("click", sampleInfer);
refresh();
setInterval(refresh, 4000);
