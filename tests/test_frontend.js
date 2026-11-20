const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../working/web/templates/index.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function createContext(responses) {
  const elements = new Map();
  const timers = new Map();
  const requests = [];
  let nextTimer = 1;
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, {
      style: {}, classList: { add() {}, remove() {} }, addEventListener() {},
      textContent: '', className: '', value: '', play() {},
      attributes: {}, setAttribute(name, value) { this.attributes[name] = value; },
    });
    return elements.get(id);
  };
  const context = vm.createContext({
    document: { getElementById: element }, console,
    FormData: class { constructor() { this.values = {}; }
      append(name, value) { this.values[name] = value; }
      get(name) { return this.values[name]; } },
    setInterval(callback) { const id = nextTimer++; timers.set(id, callback); return id; },
    clearInterval(id) { timers.delete(id); },
    async fetch(url, options) {
      requests.push(url);
      const result = responses.shift();
      if (!result) throw new Error('Unexpected request: ' + url);
      if (result.checkRequest) result.checkRequest(options);
      return { ok: result.status >= 200 && result.status < 300, status: result.status,
        async json() { if (result.invalidJson) throw new Error('Invalid JSON'); return result.body; } };
    },
  });
  vm.runInContext(source, context);
  return { context, elements, timers, requests };
}

const flush = () => new Promise(resolve => setImmediate(resolve));

(async () => {
  const rejected = createContext([{ status: 400, body: { detail: '不支援的檔案格式' } }]);
  vm.runInContext("handleFile({name: 'bad.txt'})", rejected.context);
  await flush();
  assert.equal(rejected.timers.size, 0);
  assert.deepEqual(rejected.requests, ['/upload']);
  assert.equal(rejected.elements.get('badge').className, 'badge error');
  assert.match(rejected.elements.get('pct').textContent, /HTTP 400/);

  const missing = createContext([
    { status: 200, body: { job_id: 'abc123' } },
    { status: 404, body: { detail: '找不到此任務' } },
  ]);
  vm.runInContext("handleFile({name: 'clip.mp4'})", missing.context);
  await flush();
  assert.equal(missing.timers.size, 1);
  await [...missing.timers.values()][0]();
  assert.deepEqual(missing.requests, ['/upload', '/status/abc123']);
  assert.equal(missing.timers.size, 0);
  assert.match(missing.elements.get('pct').textContent, /HTTP 404/);

  const malformed = createContext([{ status: 502, invalidJson: true }]);
  vm.runInContext("handleFile({name: 'clip.mp4'})", malformed.context);
  await flush();
  assert.equal(malformed.timers.size, 0);
  assert.match(malformed.elements.get('pct').textContent, /HTTP 502/);

  const noJob = createContext([]);
  vm.runInContext('pollStatus()', noJob.context);
  assert.equal(noJob.timers.size, 0);
  assert.equal(noJob.requests.length, 0);

  const speed = createContext([
    { status: 200, body: { job_id: 'speed123' } },
    { status: 200, body: { status: 'processing', phase: 'analysis', progress: 10,
      limit_to_source_fps: false, source_fps: 59.94, analysis_fps: 83.5, speed_revision: 0 } },
    { status: 200, body: { status: 'processing', phase: 'analysis', limit_to_source_fps: true,
      source_fps: 59.94, analysis_fps: 61.2, speed_revision: 1 },
      checkRequest(options) { assert.equal(options.method, 'POST');
        assert.deepEqual(JSON.parse(options.body), {limit_to_source_fps: true}); } },
    { status: 200, body: { status: 'processing', phase: 'analysis', limit_to_source_fps: false,
      source_fps: 59.94, analysis_fps: 70.2, speed_revision: 2 },
      checkRequest(options) { assert.deepEqual(JSON.parse(options.body), {limit_to_source_fps: false}); } },
    { status: 200, body: { status: 'processing', phase: 'export', progress: 70,
      limit_to_source_fps: false, source_fps: 59.94, analysis_fps: 70.2, speed_revision: 2 } },
  ]);
  vm.runInContext("handleFile({name: '60fps.mp4'})", speed.context);
  await flush();
  assert.match(speed.elements.get('speedInfo').textContent, /不限速/);
  await [...speed.timers.values()][0]();
  assert.match(speed.elements.get('speedInfo').textContent, /59.94 FPS/);
  assert.match(speed.elements.get('speedInfo').textContent, /83.50 FPS/);
  await vm.runInContext('toggleAnalysisSpeed()', speed.context);
  assert.equal(speed.elements.get('speedBtn').attributes['aria-pressed'], 'true');
  assert.match(speed.elements.get('speedBtn').textContent, /取消/);
  await vm.runInContext('toggleAnalysisSpeed()', speed.context);
  assert.equal(speed.elements.get('speedBtn').attributes['aria-pressed'], 'false');
  await [...speed.timers.values()][0]();
  assert.equal(speed.elements.get('speedBtn').disabled, true);
  assert.match(speed.elements.get('speedNote').textContent, /輸出保持不限速/);
  assert.equal(speed.elements.get('previewVid').playbackRate, undefined);
  const requestCount = speed.requests.length;
  await vm.runInContext('toggleAnalysisSpeed()', speed.context);
  assert.equal(speed.requests.length, requestCount);

  const failedSwitch = createContext([
    { status: 409, body: { detail: '模型分析已結束' } },
  ]);
  vm.runInContext("jobId = 'race'; renderSpeedControl({status: 'processing', phase: 'analysis', limit_to_source_fps: false})", failedSwitch.context);
  await vm.runInContext('toggleAnalysisSpeed()', failedSwitch.context);
  assert.match(failedSwitch.elements.get('speedNote').textContent, /HTTP 409/);
  assert.notEqual(failedSwitch.elements.get('badge').className, 'badge error');
  assert.equal(failedSwitch.elements.get('speedBtn').disabled, false);

  const preference = createContext([
    { status: 200, body: { job_id: 'preselected', limit_to_source_fps: true },
      checkRequest(options) { assert.equal(options.body.get('limit_to_source_fps'), 'true'); } },
  ]);
  assert.equal(preference.elements.get('speedBtn').attributes['aria-pressed'], 'false');
  await vm.runInContext('toggleAnalysisSpeed()', preference.context);
  assert.equal(preference.requests.length, 0);
  assert.equal(preference.elements.get('speedBtn').attributes['aria-pressed'], 'true');
  vm.runInContext("handleFile({name: 'short.mp4'})", preference.context);
  await flush();
  assert.equal(preference.elements.get('speedBtn').attributes['aria-pressed'], 'true');
  vm.runInContext('reset()', preference.context);
  assert.equal(preference.elements.get('speedBtn').attributes['aria-pressed'], 'true');
  await vm.runInContext('toggleAnalysisSpeed()', preference.context);
  assert.equal(preference.elements.get('speedBtn').attributes['aria-pressed'], 'false');

  const replay = createContext([]);
  vm.runInContext("renderBounces([{id: 2, frame: 72, timestamp: '00:02.88', time_sec: 2.88}])", replay.context);
  const bounceTable = replay.elements.get('bounceTableContainer').innerHTML;
  assert.match(bounceTable, /onclick="jumpTo\(2\.88\)">▶ 重播/);
  assert.match(bounceTable, /onclick="jumpTo\(2\.88, 0\.5\)">0\.5× 慢放/);
  const replayVideo = replay.elements.get('previewVid');
  const playbacks = [];
  replayVideo.play = () => playbacks.push([replayVideo.currentTime, replayVideo.playbackRate]);
  vm.runInContext('jumpTo(2.88, 0.5); jumpTo(2.88); jumpTo(0.4, 0.5);', replay.context);
  assert.deepEqual(playbacks, [[1.88, 0.5], [1.88, 1], [0, 0.5]]);

  const scoring = createContext([]);
  vm.runInContext(`renderBounces([
    {id: 1, frame: 41, timestamp: '00:01.64', time_sec: 1.64, confidence: 1.135},
    {id: 2, frame: 72, timestamp: '00:02.88', time_sec: 2.88, confidence: 0.705},
    {id: 3, frame: 110, timestamp: '00:04.40', time_sec: 4.4}
  ])`, scoring.context);
  const scoredTable = scoring.elements.get('bounceTableContainer').innerHTML;
  assert.match(scoredTable, /<th>落點分數<\/th>/);
  assert.match(scoredTable, />1\.135<\/span>/);
  assert.match(scoredTable, />0\.705<\/span>/);
  assert.match(scoredTable, /<td>-<\/td>/);

  console.log('9 frontend checks passed (4 error paths, live speed toggle/export, failed switch, pre-upload preference, bounce replay, raw bounce scores).');
})().catch(error => { console.error(error); process.exitCode = 1; });
