const state = { overview: null, trades: [], signals: [], sort: {} };
const dataPath = (name) => `../output/dashboard/${name}.json?ts=${Date.now()}`;
const qs = (id) => document.getElementById(id);

function fmtNum(value, digits = 2) {
  if (value === null || value === undefined || value === '') return '-';
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return n.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
function fmtPrice(value) {
  if (value === null || value === undefined || value === '') return '-';
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  if (n === 0) return '0';
  const abs = Math.abs(n);
  if (abs >= 1) return fmtNum(n, 2);
  const decimals = Math.max(2, Math.floor(-Math.log10(abs)) + 2);
  return n.toLocaleString('en-US', {
    maximumFractionDigits: decimals,
    minimumFractionDigits: 0,
  });
}
function fmtPct(value) { return value === null || value === undefined ? '-' : `${fmtNum(value, 2)}%`; }
function fmtU(value) { return value === null || value === undefined ? '-' : `${fmtNum(value, 2)} U`; }
function pnlClass(value) { const n = Number(value); return n > 0 ? 'pnl-pos' : n < 0 ? 'pnl-neg' : ''; }
function text(value) { return value === null || value === undefined || value === '' ? '-' : String(value); }

async function loadData() {
  const [overview, trades, signals] = await Promise.all([
    fetch(dataPath('overview')).then(r => r.json()),
    fetch(dataPath('trades')).then(r => r.json()),
    fetch(dataPath('signals')).then(r => r.json()),
  ]);
  state.overview = overview;
  state.trades = trades;
  state.signals = signals;
  renderAll();
}

function renderAll() {
  renderOverview();
  populateSignalSources();
  renderTrades();
  renderSignals();
}

function renderOverview() {
  const o = state.overview || {};
  qs('generatedAt').textContent = o.generated_at_utc || '-';
  qs('overviewSub').textContent = `${text(o.data_source || o.mode)} / ${text(o.signal_mode)} | start: ${text(o.start_time_bj)} | state: ${text(o.state_path)}`;
  const net = o.month_net_pnl_u;
  const badge = qs('healthBadge');
  badge.className = 'badge neutral';
  badge.textContent = 'WATCH';
  if (Number(net) > 0 && Number(o.month_pf || 0) >= 1) { badge.className = 'badge good'; badge.textContent = 'HEALTHY'; }
  if (Number(net) < 0 || Number(o.month_pf || 9) < 0.8) { badge.className = 'badge danger'; badge.textContent = 'RISK'; }
  const metrics = [
    ['当前持仓', `${o.blocking_positions_count ?? 0}`, `real ${o.open_positions_count ?? 0} / virtual ${o.virtual_positions_count ?? 0}`],
    ['未平仓浮盈亏', fmtU(o.open_pnl_u), `${o.open_pnl_u_known ?? 0} known positions`],
    [`${o.month || ''} 已实现`, fmtU(o.month_realized_pnl_u), `${o.month_trades ?? 0} trades`],
    [`${o.month || ''} 净收益`, fmtU(o.month_net_pnl_u), 'realized + MTM'],
    ['本月 PF', fmtNum(o.month_pf), 'gross profit / gross loss'],
    ['本月胜率', fmtPct(o.month_win_rate_pct), 'closed + MTM rows'],
    ['本月强平', `${o.month_liquidations ?? 0}`, fmtPct(o.month_liq_rate_pct)],
    ['Regime', text(o.regime_state || 'N/A'), `recovery ${text(o.recovery_signal)} / ${text(o.recovery_streak)}`],
    ['最近信号', text(o.last_signal_time_utc), '交易窗口'],
    ['最近观察', text(o.last_information_time_utc), '信息窗口'],
    ['最近预检', text(o.last_preflight_time_utc), 'market preflight'],
    ['数据来源', text(o.data_source || 'unknown'), text(o.start_time_bj)],
    ['数据生成', text(o.generated_at_utc), 'local export'],
  ];
  qs('metricGrid').innerHTML = metrics.map(([label, value, hint]) => `<div class="metric"><span>${label}</span><strong class="${label.includes('收益') || label.includes('浮盈亏') || label.includes('实现') ? pnlClass(String(value).replace(/[^-0-9.]/g,'')) : ''}">${value}</strong><small>${hint}</small></div>`).join('');
  qs('positionCount').textContent = `${(o.positions || []).length} rows`;
  renderTable('positionsTable', o.positions || [], [
    ['source','来源'], ['symbol','Symbol'], ['rank','Rank'], ['entry_time_utc','入场UTC'], ['entry_price','入场价','price'], ['current_price','当前价','price'], ['leverage','杠杆','num'], ['pnl_u','浮盈亏U','num pnl'], ['return_pct','收益率','num pct'], ['planned_exit_utc','计划退出'], ['weak_exit_checked','12H检查'], ['extreme_weak_exit_checked','4H检查']
  ]);
}

function renderTopTrades() {
  const rows = [...state.trades]
    .filter(r => r.pnl_u !== null && r.pnl_u !== undefined)
    .sort((a, b) => Number(b.pnl_u || 0) - Number(a.pnl_u || 0))
    .slice(0, 30);
  renderTable('topTradesTable', rows, [
    ['symbol','Symbol'], ['pnl_u','??','num pnl'], ['rank','??'], ['leverage','??','num'], ['max_profit_pct','????','num pct'], ['max_profit_u','????U','num pnl'], ['max_profit_day','??????','num'], ['holding_days','????','num'], ['status','??'], ['entry_time_utc','??UTC']
  ]);
}

function renderTrades() {
  renderTopTrades();
  const q = qs('tradeSearch').value.trim().toLowerCase();
  const rank = qs('tradeRank').value;
  const status = qs('tradeStatus').value;
  const liq = qs('tradeLiq').value;
  let rows = state.trades.filter(r => {
    const hay = `${r.symbol} ${r.exit_reason} ${r.strategy_component} ${r.bucket}`.toLowerCase();
    return (!q || hay.includes(q)) && (!rank || String(r.rank) === rank) && (!status || r.status === status) && (!liq || String(Boolean(r.liquidated)) === liq);
  });
  rows = rows.slice(0, 800);
  renderTable('tradesTable', rows, [
    ['month','月份'], ['symbol','Symbol'], ['rank','Rank'], ['status','状态'], ['entry_time_utc','入场UTC'], ['exit_time_utc','退出UTC'], ['entry_price','入场价','price'], ['exit_price','退出价','price'], ['gain_24h_pct','24H涨幅','num pct'], ['volume_24h_ratio_7d','量比','num'], ['leverage','杠杆','num'], ['holding_days','持有D','num'], ['exit_reason','退出原因'], ['pnl_u','收益U','num pnl'], ['net_return_pct','收益率','num pct'], ['max_profit_pct','????','num pct'], ['max_profit_u','????U','num pnl'], ['max_profit_day','??????','num'], ['mfe_pct','MFE','num pct'], ['mae_pct','MAE','num pct'], ['liquidated','强平']
  ]);
}

function populateSignalSources() {
  const select = qs('signalSource');
  const current = select.value;
  const sources = [...new Set(state.signals.map(r => r.source).filter(Boolean))].sort();
  select.innerHTML = '<option value="">全部来源</option>' + sources.map(s => `<option value="${s}">${s}</option>`).join('');
  select.value = current;
}

function renderSignals() {
  const q = qs('signalSearch').value.trim().toLowerCase();
  const rank = qs('signalRank').value;
  const pass = qs('signalPass').value;
  const source = qs('signalSource').value;
  let rows = state.signals.filter(r => {
    const hay = `${r.symbol} ${r.filter_reason} ${r.order_status} ${r.order_error}`.toLowerCase();
    return (!q || hay.includes(q)) && (!rank || String(r.rank) === rank) && (!pass || String(Boolean(r.passed)) === pass) && (!source || r.source === source);
  });
  rows = rows.slice(0, 1000);
  renderTable('signalsTable', rows, [
    ['source','来源'], ['signal_time_utc','时间UTC'], ['signal_time_bj','时间BJ'], ['symbol','Symbol'], ['rank','Rank'], ['price','价格','price'], ['gain_24h_pct','24H涨幅','num pct'], ['volume_24h_ratio_7d','量比','num'], ['passed','结果','status'], ['filter_reason','过滤原因'], ['leverage','杠杆','num'], ['planned_hold_days','计划D','num'], ['regime_state','Regime'], ['recovery_signal','Recovery'], ['order_status','订单状态'], ['order_error','错误']
  ]);
}

function renderTable(id, rows, cols) {
  const table = qs(id);
  if (!rows.length) { table.innerHTML = '<tbody><tr><td>暂无数据</td></tr></tbody>'; return; }
  table.innerHTML = `<thead><tr>${cols.map(([key,label]) => `<th data-key="${key}">${label}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${cols.map(col => cell(row, col)).join('')}</tr>`).join('')}</tbody>`;
  table.querySelectorAll('th').forEach(th => th.addEventListener('click', () => sortRows(id, th.dataset.key, cols)));
}

function cell(row, [key, label, kind = '']) {
  const value = row[key];
  if (kind.includes('price')) return `<td class="num">${fmtPrice(value)}</td>`;
  if (kind.includes('pct')) return `<td class="num">${fmtPct(value)}</td>`;
  if (kind.includes('pnl')) return `<td class="num ${pnlClass(value)}">${fmtU(value)}</td>`;
  if (kind.includes('num')) return `<td class="num">${fmtNum(value)}</td>`;
  if (kind.includes('status')) {
    const cls = value ? 'status-pass' : (row.order_status === 'failed' ? 'status-fail' : 'status-skip');
    return `<td class="${cls}">${value ? 'PASS' : 'SKIP'}</td>`;
  }
  return `<td>${text(value)}</td>`;
}

function sortRows(tableId, key, cols) {
  const dataName = tableId === 'tradesTable' ? 'trades' : tableId === 'signalsTable' ? 'signals' : null;
  if (!dataName) return;
  const dir = state.sort[tableId] === key ? -1 : 1;
  state.sort[tableId] = dir === 1 ? key : '';
  state[dataName].sort((a,b) => (a[key] > b[key] ? dir : a[key] < b[key] ? -dir : 0));
  if (tableId === 'tradesTable') renderTrades(); else renderSignals();
}

document.querySelectorAll('.nav-btn').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  btn.classList.add('active');
  qs(btn.dataset.view).classList.add('active');
}));
['tradeSearch','tradeRank','tradeStatus','tradeLiq'].forEach(id => qs(id).addEventListener('input', renderTrades));
['signalSearch','signalRank','signalPass','signalSource'].forEach(id => qs(id).addEventListener('input', renderSignals));
qs('reloadBtn').addEventListener('click', loadData);
loadData().catch(err => {
  qs('overviewSub').textContent = `读取失败：${err.message}`;
  console.error(err);
});
