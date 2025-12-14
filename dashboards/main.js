/*
 * 无线边缘联邦学习调度仪表盘前端逻辑
 * - 加载 dashboards/metrics_summary.json 与 dashboards/pareto_summary.json
 * - 使用 Chart.js 绘制训练曲线与 Pareto 散点
 * - 提供运行选择、Pareto 轴切换与简单统计视图
 */

let metricsData = null;
let paretoData = null;
let currentRun = null;

const charts = {
  accuracy: null,
  fairness: null,
  energy: null,
  selected: null,
  pareto: null,
};

const paretoAxisOptions = {
  acc: "精度 acc",
  time: "时间 time",
  fairness: "公平 fairness",
  energy: "能耗 energy",
};

function setStatus(msg, isError = false) {
  const el = document.getElementById("status-message");
  if (!el) return;
  el.textContent = msg;
  el.style.color = isError ? "#b91c1c" : "#52606d";
}

async function loadData() {
  try {
    const [mResp, pResp] = await Promise.all([
      fetch("metrics_summary.json"),
      fetch("pareto_summary.json"),
    ]);

    if (!mResp.ok) {
      throw new Error("无法加载 metrics_summary.json (请先运行 scripts/prepare_dashboard_data.py)");
    }
    metricsData = await mResp.json();

    if (pResp.ok) {
      paretoData = await pResp.json();
    } else {
      paretoData = { solutions: [] };
    }

    setStatus("数据加载完成。");
  } catch (err) {
    console.error(err);
    setStatus(`数据加载失败：${err.message}`, true);
  }
}

function initRunSelector() {
  const runs = (metricsData && Array.isArray(metricsData.runs)) ? metricsData.runs : [];
  const select = document.getElementById("run-select");
  if (!select) return;

  select.innerHTML = "";

  if (runs.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "未找到运行数据";
    select.appendChild(opt);
    return;
  }

  runs.forEach((run, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    const name = run.display_name || run.run_id || `run_${idx}`;
    opt.textContent = name;
    select.appendChild(opt);
  });

  select.addEventListener("change", () => {
    const idx = parseInt(select.value, 10);
    if (!Number.isNaN(idx) && runs[idx]) {
      setCurrentRun(runs[idx]);
    }
  });

  // 默认选中第一个运行
  setCurrentRun(runs[0]);
  select.value = "0";
}

function setCurrentRun(run) {
  currentRun = run;
  if (!run || !Array.isArray(run.metrics)) return;
  updateTrainingCharts(run);
  updateMetricsSummary(run);
  updateSelectedChart(run);
}

function ensureLineChart(chartKey, canvasId, label, color, labels, data) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const dataset = {
    label,
    data,
    borderColor: color,
    backgroundColor: color.replace("1)", "0.15)"),
    tension: 0.15,
    pointRadius: 2.5,
  };

  if (!charts[chartKey]) {
    charts[chartKey] = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [dataset],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: true },
          tooltip: { enabled: true },
        },
        scales: {
          x: {
            title: { display: true, text: "轮次" },
          },
          y: {
            beginAtZero: false,
          },
        },
      },
    });
  } else {
    const chart = charts[chartKey];
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].label = label;
    chart.update();
  }
}

function updateTrainingCharts(run) {
  const metrics = run.metrics || [];
  const rounds = metrics.map((m) => m.round);

  const accuracies = metrics.map((m) => m.accuracy ?? null);
  const fairness = metrics.map((m) => m.jain_index ?? null);
  const totalEnergy = metrics.map((m) => m.total_energy ?? null);
  const commEnergy = metrics.map((m) => m.comm_energy ?? null);
  const compEnergy = metrics.map((m) => m.comp_energy ?? null);

  // 精度
  ensureLineChart(
    "accuracy",
    "accuracy-chart",
    "测试精度",
    "rgba(37, 99, 235, 1)",
    rounds,
    accuracies,
  );

  // 公平性
  ensureLineChart(
    "fairness",
    "fairness-chart",
    "Jain 公平指数",
    "rgba(16, 185, 129, 1)",
    rounds,
    fairness,
  );

  // 能耗：同时展示总能耗、通信与计算能耗
  const ctx = document.getElementById("energy-chart");
  if (!ctx) return;

  if (!charts.energy) {
    charts.energy = new Chart(ctx, {
      type: "line",
      data: {
        labels: rounds,
        datasets: [
          {
            label: "总能耗 (通信+计算)",
            data: totalEnergy,
            borderColor: "rgba(249, 115, 22, 1)",
            backgroundColor: "rgba(249, 115, 22, 0.18)",
            tension: 0.15,
            pointRadius: 2.5,
          },
          {
            label: "通信能耗",
            data: commEnergy,
            borderColor: "rgba(59, 130, 246, 1)",
            backgroundColor: "rgba(59, 130, 246, 0.12)",
            tension: 0.15,
            borderDash: [4, 3],
            pointRadius: 2,
          },
          {
            label: "计算能耗",
            data: compEnergy,
            borderColor: "rgba(16, 185, 129, 1)",
            backgroundColor: "rgba(16, 185, 129, 0.12)",
            tension: 0.15,
            borderDash: [2, 3],
            pointRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: true },
          tooltip: { enabled: true },
        },
        scales: {
          x: { title: { display: true, text: "轮次" } },
          y: { beginAtZero: false, title: { display: true, text: "能耗 (相对单位)" } },
        },
      },
    });
  } else {
    const chart = charts.energy;
    chart.data.labels = rounds;
    chart.data.datasets[0].data = totalEnergy;
    chart.data.datasets[1].data = commEnergy;
    chart.data.datasets[2].data = compEnergy;
    chart.update();
  }
}

function updateSelectedChart(run) {
  const metrics = run.metrics || [];
  const rounds = metrics.map((m) => m.round);
  const selectedCounts = metrics.map((m) => {
    if (Array.isArray(m.selected)) return m.selected.length;
    return null;
  });

  const ctx = document.getElementById("selected-chart");
  if (!ctx) return;

  if (!charts.selected) {
    charts.selected = new Chart(ctx, {
      type: "bar",
      data: {
        labels: rounds,
        datasets: [
          {
            label: "每轮选中客户端数",
            data: selectedCounts,
            backgroundColor: "rgba(96, 165, 250, 0.7)",
            borderColor: "rgba(37, 99, 235, 1)",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          x: { title: { display: true, text: "轮次" } },
          y: { beginAtZero: true, title: { display: true, text: "客户端数量" } },
        },
      },
    });
  } else {
    const chart = charts.selected;
    chart.data.labels = rounds;
    chart.data.datasets[0].data = selectedCounts;
    chart.update();
  }
}

function updateMetricsSummary(run) {
  const metrics = run.metrics || [];
  const list = document.getElementById("metrics-summary-list");
  if (!list) return;
  list.innerHTML = "";

  if (metrics.length === 0) {
    const li = document.createElement("li");
    li.textContent = "无可用指标";
    list.appendChild(li);
    return;
  }

  const rounds = metrics.map((m) => m.round);
  const accuracies = metrics.map((m) => m.accuracy ?? 0);
  const fairness = metrics.map((m) => m.jain_index ?? 0);
  const totalEnergy = metrics.map((m) => m.total_energy ?? 0);
  const commTime = metrics.map((m) => m.comm_time ?? 0);

  const last = metrics[metrics.length - 1];

  const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);

  const items = [
    ["运行标识", run.display_name || run.run_id || "(未命名)"],
    ["轮次数", String(rounds.length)],
    ["最终精度", `${((last.accuracy ?? 0) * 100).toFixed(2)} %`],
    ["平均精度", `${(avg(accuracies) * 100).toFixed(2)} %`],
    ["平均 Jain 公平指数", avg(fairness).toFixed(3)],
    ["平均总能耗 (通信+计算)", avg(totalEnergy).toFixed(3)],
    ["平均通信时间", avg(commTime).toFixed(3)],
  ];

  for (const [k, v] of items) {
    const li = document.createElement("li");
    const spanKey = document.createElement("span");
    spanKey.className = "key";
    spanKey.textContent = k;
    const spanVal = document.createElement("span");
    spanVal.className = "value";
    spanVal.textContent = v;
    li.appendChild(spanKey);
    li.appendChild(spanVal);
    list.appendChild(li);
  }
}

function initParetoSection() {
  const solutions = paretoData && Array.isArray(paretoData.solutions) ? paretoData.solutions : [];
  const xSelect = document.getElementById("pareto-x-select");
  const ySelect = document.getElementById("pareto-y-select");

  if (!xSelect || !ySelect) return;

  xSelect.innerHTML = "";
  ySelect.innerHTML = "";

  Object.entries(paretoAxisOptions).forEach(([key, label]) => {
    const optX = document.createElement("option");
    optX.value = key;
    optX.textContent = label;
    xSelect.appendChild(optX);

    const optY = document.createElement("option");
    optY.value = key;
    optY.textContent = label;
    ySelect.appendChild(optY);
  });

  // 默认展示：能耗 vs 精度
  xSelect.value = "energy";
  ySelect.value = "acc";

  xSelect.addEventListener("change", () => updateParetoChart());
  ySelect.addEventListener("change", () => updateParetoChart());

  if (!solutions.length) {
    const detailPre = document.getElementById("pareto-detail-pre");
    if (detailPre) {
      detailPre.textContent = "未检测到 Pareto 候选数据，请先运行遗传优化脚本生成 pareto_candidates.csv。";
    }
    return;
  }

  updateParetoChart();
}

function updateParetoChart() {
  const solutions = paretoData && Array.isArray(paretoData.solutions) ? paretoData.solutions : [];
  const xSelect = document.getElementById("pareto-x-select");
  const ySelect = document.getElementById("pareto-y-select");
  const detailPre = document.getElementById("pareto-detail-pre");

  if (!xSelect || !ySelect) return;

  const xKey = xSelect.value || "energy";
  const yKey = ySelect.value || "acc";

  const dataPoints = solutions.map((s) => ({
    x: s[xKey],
    y: s[yKey],
    id: s.id,
  }));

  const ctx = document.getElementById("pareto-chart");
  if (!ctx) return;

  const labels = {
    acc: "精度 (acc)",
    time: "时间 (time)",
    fairness: "公平 (fairness)",
    energy: "能耗 (energy)",
  };

  if (!charts.pareto) {
    charts.pareto = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Pareto 候选",
            data: dataPoints,
            backgroundColor: "rgba(59, 130, 246, 0.85)",
            borderColor: "rgba(30, 64, 175, 1)",
            pointRadius: 5,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const index = ctx.dataIndex;
                const s = solutions[index];
                if (!s) return "";
                const parts = [];
                parts.push(`acc=${(s.acc * 100).toFixed(2)}%`);
                parts.push(`time=${s.time?.toFixed ? s.time.toFixed(3) : s.time}`);
                parts.push(`fairness=${s.fairness?.toFixed ? s.fairness.toFixed(3) : s.fairness}`);
                parts.push(`energy=${s.energy?.toFixed ? s.energy.toFixed(3) : s.energy}`);
                parts.push(`Top-K=${s.selection_top_k}`);
                return parts.join(" | ");
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: labels[xKey] || xKey },
          },
          y: {
            title: { display: true, text: labels[yKey] || yKey },
          },
        },
        onClick: (event, elements) => {
          if (!elements.length || !detailPre) return;
          const idx = elements[0].index;
          const s = solutions[idx];
          if (!s) return;
          const view = {
            id: s.id,
            acc: s.acc,
            time: s.time,
            fairness: s.fairness,
            energy: s.energy,
            energy_w: s.energy_w,
            channel_w: s.channel_w,
            data_w: s.data_w,
            fair_w: s.fair_w,
            bwcost_w: s.bwcost_w,
            selection_top_k: s.selection_top_k,
            hysteresis: s.hysteresis,
            staleness_alpha: s.staleness_alpha,
          };
          detailPre.textContent = JSON.stringify(view, null, 2);
        },
      },
    });
  } else {
    const chart = charts.pareto;
    chart.data.datasets[0].data = dataPoints;
    chart.options.scales.x.title.text = labels[xKey] || xKey;
    chart.options.scales.y.title.text = labels[yKey] || yKey;
    chart.update();
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadData();
  if (metricsData) {
    initRunSelector();
  }
  if (paretoData) {
    initParetoSection();
  }
});
