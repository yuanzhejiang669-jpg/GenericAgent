import os, sys, threading, queue, time, json, re, random, locale
os.environ.setdefault('GA_LANG', 'zh' if any(k in (locale.getlocale()[0] or '').lower() for k in ('zh', 'chinese')) else 'en')
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
elif hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(errors='replace')
if sys.stderr is None: sys.stderr = open(os.devnull, "w")
elif hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(errors='replace')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llmcore import reload_mykeys, LLMSession, ToolClient, ClaudeSession, MixinSession, NativeToolClient, NativeClaudeSession, NativeOAISession
from agent_loop import agent_runner_loop
from ga import GenericAgentHandler, smart_format, get_global_memory, format_error, consume_file

script_dir = os.path.dirname(os.path.abspath(__file__))
def load_tool_schema(suffix=''):
    global TOOLS_SCHEMA
    TS = open(os.path.join(script_dir, f'assets/tools_schema{suffix}.json'), 'r', encoding='utf-8').read()
    TOOLS_SCHEMA = json.loads(TS if os.name == 'nt' else TS.replace('powershell', 'bash'))
load_tool_schema()

lang_suffix = '_en' if os.environ.get('GA_LANG', '') == 'en' else ''
mem_dir = os.path.join(script_dir, 'memory')
if not os.path.exists(mem_dir): os.makedirs(mem_dir)
mem_txt = os.path.join(mem_dir, 'global_mem.txt')
if not os.path.exists(mem_txt): open(mem_txt, 'w', encoding='utf-8').write('# [Global Memory - L2]\n')
mem_insight = os.path.join(mem_dir, 'global_mem_insight.txt')
if not os.path.exists(mem_insight):
    t = os.path.join(script_dir, f'assets/global_mem_insight_template{lang_suffix}.txt')
    open(mem_insight, 'w', encoding='utf-8').write(open(t, encoding='utf-8').read() if os.path.exists(t) else '')
cdp_cfg = os.path.join(script_dir, 'assets/tmwd_cdp_bridge/config.js')
if not os.path.exists(cdp_cfg):
    try:
        os.makedirs(os.path.dirname(cdp_cfg), exist_ok=True)
        open(cdp_cfg, 'w', encoding='utf-8').write(f"const TID = '__ljq_{hex(random.randint(0, 99999999))[2:8]}';")
    except Exception as e: print(f'[WARN] CDP config init failed: {e} — advanced web features (tmwebdriver) will be unavailable.')

def get_system_prompt():
    with open(os.path.join(script_dir, f'assets/sys_prompt{lang_suffix}.txt'), 'r', encoding='utf-8') as f: prompt = f.read()
    prompt += f"\nToday: {time.strftime('%Y-%m-%d %a')}\n"
    prompt += get_global_memory()
    return prompt

class GeneraticAgent:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, 'temp'), exist_ok=True)
        self.lock = threading.Lock()
        self.task_dir = None
        self.history = []
        self.task_queue = queue.Queue() 
        self.is_running = False; self.stop_sig = False
        self.llm_no = 0;  self.inc_out = False
        self.handler = None; self.verbose = True
        self.load_llm_sessions()

    def load_llm_sessions(self):
        mykeys, changed = reload_mykeys()
        if not changed and hasattr(self, 'llmclients'): return
        try: oldhistory = self.llmclient.backend.history
        except: oldhistory = None
        llm_sessions = []
        for k, cfg in mykeys.items():
            if not any(x in k for x in ['api', 'config', 'cookie']): continue
            try:
                if 'native' in k and 'claude' in k: llm_sessions += [NativeToolClient(NativeClaudeSession(cfg=cfg))]
                elif 'native' in k and 'oai' in k: llm_sessions += [NativeToolClient(NativeOAISession(cfg=cfg))]
                elif 'claude' in k: llm_sessions += [ToolClient(ClaudeSession(cfg=cfg))]
                elif 'oai' in k: llm_sessions += [ToolClient(LLMSession(cfg=cfg))]
                elif 'mixin' in k: llm_sessions += [{'mixin_cfg': cfg}]
            except: pass
        for i, s in enumerate(llm_sessions):
            if isinstance(s, dict) and 'mixin_cfg' in s:
                try:
                    mixin = MixinSession(llm_sessions, s['mixin_cfg'])
                    if isinstance(mixin._sessions[0], (NativeClaudeSession, NativeOAISession)): llm_sessions[i] = NativeToolClient(mixin)
                    else: llm_sessions[i] = ToolClient(mixin)
                except Exception as e: print(f'[WARN] Failed to init MixinSession with cfg {s["mixin_cfg"]}: {e}')
        self.llmclients = llm_sessions
        self.llmclient = self.llmclients[self.llm_no%len(self.llmclients)]
        if oldhistory: self.llmclient.backend.history = oldhistory
    
    def next_llm(self, n=-1):
        self.load_llm_sessions()
        self.llm_no = ((self.llm_no + 1) if n < 0 else n) % len(self.llmclients)
        lastc = self.llmclient
        self.llmclient = self.llmclients[self.llm_no]
        try: self.llmclient.backend.history = lastc.backend.history
        except: raise Exception('[ERROR] BAD Mixin config: Check your mykey.py')
        self.llmclient.last_tools = ''
        name = self.get_llm_name(model=True)
        if 'glm' in name or 'minimax' in name or 'kimi' in name: load_tool_schema('_cn')
        else: load_tool_schema()
    def list_llms(self): 
        self.load_llm_sessions()
        return [(i, self.get_llm_name(b), i == self.llm_no) for i, b in enumerate(self.llmclients)]
    def get_llm_name(self, b=None, model=False):
        b = self.llmclient if b is None else b
        if isinstance(b, dict): return 'BADCONFIG_MIXIN'
        if model: return b.backend.model.lower()
        return f"{type(b.backend).__name__}/{b.backend.name}"

    def abort(self):
        if not self.is_running: return
        print('Abort current task...')
        self.stop_sig = True
        if self.task_dir: open(os.path.join(self.task_dir, '_stop'), 'w', encoding='utf-8').write('1')
        if self.handler is not None: self.handler.code_stop_signal.append(1)
            
    def put_task(self, query, source="user", images=None):
        display_queue = queue.Queue()
        self.task_queue.put({"query": query, "source": source, "images": images or [], "output": display_queue})
        return display_queue

    # i know it is dangerous, but raw_query is dangerous enough it doesn't enlarge
    def _handle_slash_cmd(self, raw_query, display_queue):
        if not raw_query.startswith('/'): return raw_query
        if _sm := re.match(r'/session\.(\w+)=(.*)', raw_query.strip()):
            k, v = _sm.group(1), _sm.group(2)
            vfile = os.path.join(script_dir, 'temp', v)
            if os.path.isfile(vfile): v = open(vfile, encoding='utf-8').read().strip()
            try: v = json.loads(v)  # cover number parsing
            except (json.JSONDecodeError, ValueError): pass
            setattr(self.llmclient.backend, k, v)
            display_queue.put({'done': smart_format(f"✅ session.{k} = {repr(v)}", max_str_len=500), 'source': 'system'})
            return None
        if raw_query.strip() == '/resume':
            return r'用re.findall(r"<history>\\n\[(?:USER\|Agent)\].*?</history>", content, re.DOTALL) 扫temp/model_responses/下时间最近的10个文件(除本PID)，取每文件最后一个匹配(注意JSON里换行是字面\\n)作为该会话内容，按mtime倒序，每个用一句话总结聊了什么让我选择；选定后再简单读该文件末尾作为聊天基础'
        return raw_query

    def run(self):
        while True:
            task = self.task_queue.get()
            raw_query, source, images, display_queue = task["query"], task["source"], task.get("images") or [], task["output"]
            raw_query = self._handle_slash_cmd(raw_query, display_queue)
            if raw_query is None:
                self.task_queue.task_done(); continue
            self.is_running = True
            rquery = smart_format(raw_query.replace('\n', ' '), max_str_len=200)
            self.history.append(f"[USER]: {rquery}")
            
            sys_prompt = get_system_prompt() + getattr(self.llmclient.backend, 'extra_sys_prompt', '')
            script_dir = os.path.dirname(os.path.abspath(__file__))
            handler = GenericAgentHandler(self, self.history, os.path.join(script_dir, 'temp'))
            if self.handler and 'key_info' in self.handler.working: 
                ki = re.sub(r'\n\[SYSTEM\] 此为.*?工作记忆[。\n]*', '', self.handler.working['key_info'])  # 去旧
                handler.working['key_info'] = ki
                handler.working['passed_sessions'] = ps = self.handler.working.get('passed_sessions', 0) + 1
                if ps > 0: handler.working['key_info'] += f'\n[SYSTEM] 此为 {ps} 个对话前设置的key_info，若已在新任务，先更新或清除工作记忆。\n'
            self.handler = handler
            # although new handler, the **full** history is in llmclient, so it is full history!
            gen = agent_runner_loop(self.llmclient, sys_prompt, raw_query, 
                                handler, TOOLS_SCHEMA, max_turns=70, verbose=self.verbose)
            try:
                full_resp = ""; last_pos = 0
                for chunk in gen:
                    if consume_file(self.task_dir, '_stop'): self.abort() 
                    if self.stop_sig: break
                    full_resp += chunk
                    if len(full_resp) - last_pos > 50 or 'LLM Running' in chunk:
                        display_queue.put({'next': full_resp[last_pos:] if self.inc_out else full_resp, 'source': source})
                        last_pos = len(full_resp)
                if self.inc_out and last_pos < len(full_resp): display_queue.put({'next': full_resp[last_pos:], 'source': source})
                if '</summary>' in full_resp: full_resp = full_resp.replace('</summary>', '</summary>\n\n')
                if '</file_content>' in full_resp: full_resp = re.sub(r'<file_content>\s*(.*?)\s*</file_content>', r'\n````\n<file_content>\n\1\n</file_content>\n````', full_resp, flags=re.DOTALL)                
                display_queue.put({'done': full_resp, 'source': source})
                self.history = handler.history_info
            except Exception as e:
                print(f"Backend Error: {format_error(e)}")
                display_queue.put({'done': full_resp + f'\n```\n{format_error(e)}\n```', 'source': source})
            finally:
                if self.stop_sig:
                    print('User aborted the task.')
                    #with self.task_queue.mutex: self.task_queue.queue.clear()
                self.is_running = self.stop_sig = False
                self.task_queue.task_done()
                if self.handler is not None: self.handler.code_stop_signal.append(1)



_bridge_session_locks = {}
_bridge_session_locks_guard = threading.Lock()
_bridge_token = os.environ.get('BROWSER_BRIDGE_TOKEN', '')


def _bridge_json_response(handler, payload, status=200):
    data = json.dumps(payload, ensure_ascii=False, default=str).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _bridge_read_json(handler):
    length = int(handler.headers.get('Content-Length') or 0)
    if length > 1024 * 1024: raise ValueError('request body too large')
    if length <= 0: return {}
    raw = handler.rfile.read(length).decode('utf-8', errors='replace')
    return json.loads(raw) if raw.strip() else {}


def _bridge_excerpt(text, limit=4000):
    if not text: return ''
    if len(text) <= limit: return text
    half = max(1, limit // 2)
    return text[:half] + '\n...[truncated]...\n' + text[-half:]


def _bridge_safe_read(path):
    try: return open(path, encoding='utf-8', errors='replace').read()
    except Exception: return ''


def _bridge_clean_session_id(session_id):
    session_id = str(session_id or '').strip()
    if session_id in {'.', '..'}: return None
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', session_id): return None
    return session_id


def _bridge_task_dir(session_id):
    temp_dir = os.path.realpath(os.path.join(script_dir, 'temp'))
    task_dir = os.path.realpath(os.path.join(temp_dir, session_id))
    if os.path.commonpath([temp_dir, task_dir]) != temp_dir: raise ValueError('invalid session path')
    return task_dir


def _bridge_output_files(task_dir):
    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem.replace('output', '', 1)
        if suffix == '': return (0, os.path.getmtime(path))
        try: return (int(suffix), os.path.getmtime(path))
        except ValueError: return (10**9, os.path.getmtime(path))
    try: return sorted([os.path.join(task_dir, f) for f in os.listdir(task_dir) if re.fullmatch(r'output\d*\.txt', f)], key=sort_key)
    except Exception: return []


def _bridge_read_pid(task_dir):
    try: return int(open(os.path.join(task_dir, 'browser_bridge.pid'), encoding='utf-8').read().strip())
    except Exception: return None


def _bridge_read_state(task_dir):
    try: return json.loads(open(os.path.join(task_dir, 'browser_bridge.state.json'), encoding='utf-8').read())
    except Exception: return {}


def _bridge_write_state(task_dir, state):
    open(os.path.join(task_dir, 'browser_bridge.state.json'), 'w', encoding='utf-8').write(json.dumps(state, ensure_ascii=False, indent=2) + '\n')


def _bridge_process_alive(pid):
    if pid is None: return False
    try:
        if os.name != 'nt':
            os.kill(pid, 0)
            return True
        import subprocess
        result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'], capture_output=True, text=True, encoding='utf-8', errors='replace', check=False)
        return any(row.split(',')[1].strip('"') == str(pid) for row in (result.stdout or '').splitlines() if row.strip() and not row.startswith('INFO:'))
    except Exception:
        return False


def _bridge_interrupt(task_dir):
    path = os.path.join(task_dir, 'interrupt.json')
    try:
        payload = json.loads(open(path, encoding='utf-8').read())
    except Exception:
        return None
    if not isinstance(payload, dict): return None
    data = payload.get('data') if isinstance(payload.get('data'), dict) else payload
    question = data.get('question')
    if not isinstance(question, str) or not question.strip(): return None
    candidates = data.get('candidates') if isinstance(data.get('candidates'), list) else []
    return {
        'status': payload.get('status', 'INTERRUPT'),
        'intent': payload.get('intent', 'HUMAN_INTERVENTION'),
        'question': question,
        'candidates': [str(x) for x in candidates],
    }


def _bridge_session_status(session_id):
    clean_id = _bridge_clean_session_id(session_id)
    if not clean_id:
        return {'task_id': session_id, 'status': 'invalid_task_id', 'exists': False, 'has_output': False}
    task_dir = _bridge_task_dir(clean_id)
    exists = os.path.isdir(task_dir)
    pid = _bridge_read_pid(task_dir) if exists else None
    process_alive = _bridge_process_alive(pid)
    output_files = _bridge_output_files(task_dir) if exists else []
    latest_output_path = output_files[-1] if output_files else None
    latest_output = _bridge_safe_read(latest_output_path) if latest_output_path else ''
    stdout_log_path = os.path.join(task_dir, 'stdout.log') if exists else None
    stderr_log_path = os.path.join(task_dir, 'stderr.log') if exists else None
    stdout_text = _bridge_safe_read(stdout_log_path) if stdout_log_path else ''
    stderr_text = _bridge_safe_read(stderr_log_path) if stderr_log_path else ''
    reply_path = os.path.join(task_dir, 'reply.txt') if exists else None
    interrupt_path = os.path.join(task_dir, 'interrupt.json') if exists else None
    interrupt_info = _bridge_interrupt(task_dir) if exists else None
    state = _bridge_read_state(task_dir) if exists else {}
    expected_output_file_count = state.get('reply_expected_output_file_count')
    waiting_for_reply = bool(process_alive and output_files and '[ROUND END]' in latest_output and not os.path.exists(reply_path) and not (isinstance(expected_output_file_count, int) and len(output_files) < expected_output_file_count))
    if not exists: status = 'missing'
    elif waiting_for_reply or interrupt_info: status = 'waiting_for_reply'
    elif process_alive: status = 'running'
    elif output_files and '[ROUND END]' in latest_output: status = 'completed'
    elif output_files: status = 'completed'
    else: status = 'failed' if stderr_text else 'empty'
    return {
        'task_id': clean_id,
        'task_dir': task_dir,
        'exists': exists,
        'status': status,
        'latest_output_path': latest_output_path,
        'latest_output_excerpt': _bridge_excerpt(latest_output),
        'stdout_log_path': stdout_log_path if stdout_log_path and os.path.exists(stdout_log_path) else None,
        'stdout_excerpt': _bridge_excerpt(stdout_text),
        'stderr_log_path': stderr_log_path if stderr_log_path and os.path.exists(stderr_log_path) else None,
        'stderr_excerpt': _bridge_excerpt(stderr_text),
        'output_file_count': len(output_files),
        'waiting_for_reply': status == 'waiting_for_reply',
        'reply_file_exists': bool(reply_path and os.path.exists(reply_path)),
        'reply_in_flight': bool(reply_path and os.path.exists(reply_path) and process_alive),
        'reply_expected_output_file_count': state.get('reply_expected_output_file_count'),
        'interrupt_pending': bool(interrupt_info),
        'interrupt_kind': interrupt_info.get('status') if interrupt_info else None,
        'interrupt_intent': interrupt_info.get('intent') if interrupt_info else None,
        'question': interrupt_info.get('question') if interrupt_info else None,
        'candidates': interrupt_info.get('candidates') if interrupt_info else [],
        'interrupt_path': interrupt_path if interrupt_path and os.path.exists(interrupt_path) else None,
        'pid_file_path': os.path.join(task_dir, 'browser_bridge.pid') if exists else None,
        'pid': pid,
        'process_alive': process_alive,
        'timed_out': False,
        'has_output': bool(output_files),
    }


def _bridge_prepare_task_dir(task_dir):
    os.makedirs(task_dir, exist_ok=True)
    for name in ('input.txt', 'reply.txt', '_stop', 'stdout.log', 'stderr.log', 'browser_bridge.pid', 'browser_bridge.state.json', 'interrupt.json'):
        path = os.path.join(task_dir, name)
        if os.path.exists(path): os.remove(path)
    for name in os.listdir(task_dir):
        if re.fullmatch(r'output\d*\.txt', name): os.remove(os.path.join(task_dir, name))


def _bridge_launch_session(payload):
    import subprocess, platform
    session_id = _bridge_clean_session_id(payload.get('session_id'))
    prompt = str(payload.get('prompt') or '').strip()
    if not session_id: return {'status': 'invalid_task_id', 'accepted': False}, 400
    if not prompt: return {'status': 'invalid_prompt', 'accepted': False}, 400
    try: llm_no = int(payload.get('llm_no') or 0)
    except (TypeError, ValueError): return {'status': 'invalid_llm_no', 'accepted': False}, 400
    with _bridge_session_locks_guard:
        lock = _bridge_session_locks.setdefault(session_id, threading.Lock())
    with lock:
        task_dir = _bridge_task_dir(session_id)
        current = _bridge_session_status(session_id)
        if current.get('process_alive'):
            current.update({'status': 'task_conflict', 'accepted': False, 'conflict_reason': f'task {session_id} is already active'})
            return current, 409
        _bridge_prepare_task_dir(task_dir)
        open(os.path.join(task_dir, 'input.txt'), 'w', encoding='utf-8').write(prompt)
        stdout_path = os.path.join(task_dir, 'stdout.log')
        stderr_path = os.path.join(task_dir, 'stderr.log')
        cmd = [sys.executable, os.path.abspath(__file__), '--task', session_id, '--llm_no', str(llm_no)]
        stdout_handle = open(stdout_path, 'w', encoding='utf-8')
        stderr_handle = open(stderr_path, 'w', encoding='utf-8')
        try:
            env = os.environ.copy()
            env.pop('BROWSER_BRIDGE_TOKEN', None)
            proc = subprocess.Popen(cmd, cwd=script_dir,
                creationflags=0x08000000 if platform.system() == 'Windows' else 0,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env)
        finally:
            stdout_handle.close()
            stderr_handle.close()
        open(os.path.join(task_dir, 'browser_bridge.pid'), 'w', encoding='utf-8').write(str(proc.pid))
        status = _bridge_session_status(session_id)
        status.update({'status': 'queued' if status['status'] in {'empty', 'running'} else status['status'], 'accepted': True})
        return status, 200


def serve_bridge(host, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import unquote, urlparse
    if host not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('bridge host must be loopback')

    class BridgeHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args): return

        def _path(self): return urlparse(self.path).path

        def do_GET(self):
            path = self._path()
            if path == '/bridge/health':
                return _bridge_json_response(self, {'status': 'ok', 'pid': os.getpid(), 'project_dir': script_dir})
            if path == '/bridge/capabilities':
                return _bridge_json_response(self, {'status': 'ok', 'tools': [t.get('function', {}).get('name') for t in TOOLS_SCHEMA], 'endpoints': ['/bridge/health', '/bridge/capabilities', '/bridge/sessions']})
            if path == '/bridge/sessions':
                return _bridge_json_response(self, {'status': 'ok', 'sessions': []})
            m = re.fullmatch(r'/bridge/sessions/([^/]+)', path)
            if m: return _bridge_json_response(self, _bridge_session_status(unquote(m.group(1))))
            return _bridge_json_response(self, {'status': 'missing', 'path': path}, 404)

        def do_POST(self):
            path = self._path()
            if not _bridge_token or self.headers.get('X-Bridge-Token') != _bridge_token:
                return _bridge_json_response(self, {'status': 'unauthorized'}, 403)
            try: payload = _bridge_read_json(self)
            except ValueError as exc: return _bridge_json_response(self, {'status': 'invalid_request', 'error': str(exc)}, 413)
            except Exception as exc: return _bridge_json_response(self, {'status': 'invalid_json', 'error': str(exc)}, 400)
            if path == '/bridge/capabilities/refresh':
                return _bridge_json_response(self, {'status': 'ok', 'tools': [t.get('function', {}).get('name') for t in TOOLS_SCHEMA], 'refreshed': True})
            if path == '/bridge/sessions':
                result, status = _bridge_launch_session(payload)
                return _bridge_json_response(self, result, status)
            m = re.fullmatch(r'/bridge/sessions/([^/]+)/(reply|stop)', path)
            if not m: return _bridge_json_response(self, {'status': 'missing', 'path': path}, 404)
            session_id, action = unquote(m.group(1)), m.group(2)
            clean_id = _bridge_clean_session_id(session_id)
            if not clean_id: return _bridge_json_response(self, {'status': 'invalid_task_id'}, 400)
            task_dir = _bridge_task_dir(clean_id)
            if action == 'reply':
                reply = str(payload.get('reply') or '').strip()
                if not reply: return _bridge_json_response(self, {'status': 'invalid_reply', 'accepted': False}, 400)
                os.makedirs(task_dir, exist_ok=True)
                _bridge_write_state(task_dir, {'reply_expected_output_file_count': len(_bridge_output_files(task_dir)) + 1})
                open(os.path.join(task_dir, 'reply.txt'), 'w', encoding='utf-8').write(reply)
                return _bridge_json_response(self, {'status': 'accepted', 'accepted': True, 'reply_accepted': True, 'reply_written': True})
            os.makedirs(task_dir, exist_ok=True)
            open(os.path.join(task_dir, '_stop'), 'w', encoding='utf-8').write('1')
            return _bridge_json_response(self, {'status': 'accepted', 'stop_requested': True, 'stop_file_written': True})

    server = ThreadingHTTPServer((host, port), BridgeHandler)
    print(f'[Bridge] listening on http://{host}:{port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', metavar='IODIR', help='一次性任务模式(文件IO)')
    parser.add_argument('--reflect', metavar='SCRIPT', help='反射模式：加载监控脚本，check()触发时发任务')
    parser.add_argument('--input', help='prompt')
    parser.add_argument('--llm_no', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--bg', action='store_true', help='popen, print PID, exit')
    parser.add_argument('--serve-bridge', action='store_true', help='run browser bridge HTTP service')
    parser.add_argument('--bridge-host', default='127.0.0.1')
    parser.add_argument('--bridge-port', type=int, default=18561)
    args = parser.parse_args()

    if args.serve_bridge:
        serve_bridge(args.bridge_host, args.bridge_port)
        sys.exit(0)

    if args.bg:
        import subprocess, platform
        cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in sys.argv[1:] if a != '--bg']
        d = os.path.join(script_dir, f'temp/{args.task}'); os.makedirs(d, exist_ok=True)
        p = subprocess.Popen(cmd, cwd=script_dir,
            creationflags=0x08000000 if platform.system() == 'Windows' else 0,
            stdout=open(os.path.join(d, 'stdout.log'), 'w', encoding='utf-8'),
            stderr=open(os.path.join(d, 'stderr.log'), 'w', encoding='utf-8'))
        print(p.pid); sys.exit(0)

    agent = GeneraticAgent()
    agent.next_llm(args.llm_no)
    agent.verbose = args.verbose
    threading.Thread(target=agent.run, daemon=True).start()

    if args.task:
        agent.task_dir = d = os.path.join(script_dir, f'temp/{args.task}'); nround = ''
        infile = os.path.join(d, 'input.txt')
        if args.input:
            os.makedirs(d, exist_ok=True)
            import glob; [os.remove(f) for f in glob.glob(os.path.join(d, 'output*.txt'))]
            with open(infile, 'w', encoding='utf-8') as f: f.write(args.input)
        with open(infile, encoding='utf-8') as f: raw = f.read()
        while True:
            dq = agent.put_task(raw, source='task')
            while 'done' not in (item := dq.get(timeout=120)): 
                if 'next' in item and random.random() < 0.95:  # 概率写一次中间结果
                    with open(f'{d}/output{nround}.txt', 'w', encoding='utf-8') as f: f.write(item.get('next', ''))
            with open(f'{d}/output{nround}.txt', 'w', encoding='utf-8') as f: f.write(item['done'] + '\n\n[ROUND END]\n')
            if consume_file(d, '_stop'): break
            for _ in range(300):  # 等reply.txt，10分钟超时
                time.sleep(2)
                if consume_file(d, '_stop'):
                    raw = ''
                    break
                if (raw := consume_file(d, 'reply.txt')): break
            else: break
            if not raw: break
            nround = nround + 1 if isinstance(nround, int) else 1
    elif args.reflect:
        import importlib.util
        spec = importlib.util.spec_from_file_location('reflect_script', args.reflect)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        _mt = os.path.getmtime(args.reflect)
        print(f'[Reflect] loaded {args.reflect}')
        while True:
            if os.path.getmtime(args.reflect) != _mt:
                try: spec.loader.exec_module(mod); _mt = os.path.getmtime(args.reflect); print('[Reflect] reloaded')
                except Exception as e: print(f'[Reflect] reload error: {e}')
            time.sleep(getattr(mod, 'INTERVAL', 5))
            try: task = mod.check()
            except Exception as e: 
                print(f'[Reflect] check() error: {e}'); continue
            if task is None: continue
            print(f'[Reflect] triggered: {task[:80]}')
            dq = agent.put_task(task, source='reflect')
            try:
                while 'done' not in (item := dq.get(timeout=120)): pass
                result = item['done']
                print(result)
            except Exception as e:
                if getattr(mod, 'ONCE', False): raise
                print(f'[Reflect] drain error: {e}'); result = f'[ERROR] {e}'
            log_dir = os.path.join(script_dir, 'temp/reflect_logs'); os.makedirs(log_dir, exist_ok=True)
            script_name = os.path.splitext(os.path.basename(args.reflect))[0]
            open(os.path.join(log_dir, f'{script_name}_{datetime.now():%Y-%m-%d}.log'), 'a', encoding='utf-8').write(f'[{datetime.now():%m-%d %H:%M}]\n{result}\n\n')
            if (on_done := getattr(mod, 'on_done', None)):
                try: on_done(result)
                except Exception as e: print(f'[Reflect] on_done error: {e}')
            if getattr(mod, 'ONCE', False): print('[Reflect] ONCE=True, exiting.'); break
    else:
        try: import readline
        except Exception: pass
        agent.inc_out = True
        while True:
            q = input('> ').strip()
            if not q: continue
            try:
                dq = agent.put_task(q, source='user')
                while True:
                    item = dq.get()
                    if 'next' in item: print(item['next'], end='', flush=True)
                    if 'done' in item: print(); break
            except KeyboardInterrupt:
                agent.abort()
                print('\n[Interrupted]')
