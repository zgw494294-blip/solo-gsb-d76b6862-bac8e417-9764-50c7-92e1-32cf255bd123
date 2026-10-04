"""播放会话定位（seek）：前跳/倒带到指定时刻并同步切换显示字幕。

覆盖：
- 定位事件语义：跳过区间不补发；先为旧游标可见 active 逐条 exit，再为
  目标 active 逐条 enter；两边都活动者也退出再进入；事件均标记目标时刻，
  同类按会话字幕顺序；同刻定位成对 exit/enter；
- 隐藏集合参与两端快照；定位不改快照、隐藏集合或接纳规则；原推进仍禁止
  倒退（422），定位允许任意非负时刻；
- 补取通道：未确认队列与确认边界保留，新事件接续 seq；倒带仍按 seq 补取；
  容量不足 503 时游标、队列、seq、计时段、播放状态均不变，0 事件不占名额；
- 自动播放：idle 定位仍 idle；running/paused/blocked 定位后 paused，
  旧驱动段不得推进，resume 从新游标重新计时、不追赶；
- HTTP：严格整数拒绝布尔/字符串/小数，负数 422；未知/已结束会话 404；
  响应含游标、源版本、active、events、status。
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app import main as main_module  # noqa: E402
from app.main import app  # noqa: E402
from app.playback import (  # noqa: E402
    AUTOPLAY_BLOCKED,
    AUTOPLAY_PAUSED,
    PlaybackManager,
)
from app.storage import SceneRecord  # noqa: E402


def scene_payload(scene_id="scene-seek", subtitles=None, track_count=2):
    if subtitles is None:
        # s1[0,1000) s2[800,3000)锁2 s3[1000,2500)：分轨 1/2/1
        subtitles = [
            {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000,
             "locked_track": 2},
            {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 2500},
        ]
    resp = TestClient(app).put(
        f"/scenes/{scene_id}",
        json={
            "track_count": track_count,
            "expected_revision": 0,
            "subtitles": subtitles,
        },
    )
    assert resp.status_code == 200


def make_session(client, scene_id="scene-seek", **kwargs):
    scene_payload(scene_id=scene_id, **kwargs)
    resp = client.post("/sessions", json={"scene_id": scene_id})
    assert resp.status_code == 200
    return resp.json()["session_id"]


def seek(client, sid, time_ms):
    return client.post(f"/sessions/{sid}/seek", json={"time_ms": time_ms})


def advance(client, sid, time_ms, path="advance"):
    return client.post(f"/sessions/{sid}/{path}", json={"time_ms": time_ms})


def open_channel(client, sid):
    return client.post(f"/sessions/{sid}/channel")


def fetch(client, sid, ack_seq):
    return client.post(
        f"/sessions/{sid}/channel/fetch", json={"ack_seq": ack_seq}
    )


def set_hidden(client, sid, speakers):
    return client.post(
        f"/sessions/{sid}/hidden_speakers",
        json={"hidden_speakers": speakers},
    )


def autoplay(client, sid, action):
    method = client.get if action == "status" else client.post
    suffix = "" if action == "status" else f"/{action}"
    return method(f"/sessions/{sid}/autoplay{suffix}")


def _triples(events):
    return [(e["time_ms"], e["type"], e["id"]) for e in events]


# ---------------------------------------------------------------------------
# 基本定位事件语义（无通道）
# ---------------------------------------------------------------------------


def test_seek_forward_skips_interval_no_replay():
    """前跳：跳过区间不补发，只比对两个时刻的可见活动快照。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-fwd")

    # 先到 900：s1、s2 活动
    body = advance(client, sid, 900).json()
    assert [a["id"] for a in body["active"]] == ["s1", "s2"]

    # 直接定位到 3000：跳过的 1000 进入/退出、2500 退出等均不补发。
    # 旧 active s1/s2 逐条 exit；目标 3000（右开）无 active，无 enter。
    resp = seek(client, sid, 3000)
    assert resp.status_code == 200
    body = resp.json()
    assert body["cursor_ms"] == 3000
    assert body["source_revision"] == 1
    assert body["active"] == []
    assert body["status"] == "idle"
    assert _triples(body["events"]) == [
        (3000, "exit", "s1"),
        (3000, "exit", "s2"),
    ]


def test_seek_forward_exits_old_and_enters_target():
    """前跳到中间时刻：原 active 退出，目标 active 进入（含两边都活动者）。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-mid")
    advance(client, sid, 900)  # s1、s2 活动

    # 定位到 2000：s1 不再活动（exit），s2 两边都活动（exit+enter），
    # s3 目标活动但原不活动（enter）。同类按会话字幕顺序：s1 在 s2 前，
    # s2 在 s3 前。
    body = seek(client, sid, 2000).json()
    assert _triples(body["events"]) == [
        (2000, "exit", "s1"),
        (2000, "exit", "s2"),
        (2000, "enter", "s2"),
        (2000, "enter", "s3"),
    ]
    assert [(a["id"], a["track"]) for a in body["active"]] == [
        ("s2", 2),
        ("s3", 1),
    ]


def test_seek_backward_reenters_active_at_target():
    """倒带：原 active 退出，目标时刻的历史 active 重新进入。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-back")
    advance(client, sid, 2000)  # s2、s3 活动

    # 倒带回 500：s2/s3 退出；s1 进入（s2 在 500 不活动）
    body = seek(client, sid, 500).json()
    assert body["cursor_ms"] == 500
    assert _triples(body["events"]) == [
        (500, "exit", "s2"),
        (500, "exit", "s3"),
        (500, "enter", "s1"),
    ]
    assert [a["id"] for a in body["active"]] == ["s1"]

    # 倒带后原推进规则不变：仍禁止倒退
    resp = advance(client, sid, 499)
    assert resp.status_code == 422
    # 从 500 正常前推到 800：窗口 (500,800] 正常补发 s2 进入
    body = client.post(
        f"/sessions/{sid}/events", json={"time_ms": 800}
    ).json()
    assert _triples(body["events"]) == [(800, "enter", "s2")]


def test_seek_to_same_cursor_reemits_exit_enter_pairs():
    """定位到当前游标：两边快照相同，每个 active 仍退出再进入。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-same")
    advance(client, sid, 2000)  # s2、s3 活动

    body = seek(client, sid, 2000).json()
    assert body["cursor_ms"] == 2000
    assert _triples(body["events"]) == [
        (2000, "exit", "s2"),
        (2000, "exit", "s3"),
        (2000, "enter", "s2"),
        (2000, "enter", "s3"),
    ]
    assert [a["id"] for a in body["active"]] == ["s2", "s3"]


def test_seek_zero_to_zero_reemits_for_zero_ms_active():
    """初始游标 0 定位到 0：0 时刻已活动的字幕（创建快照确定、推进不补发
    进入）在定位里仍按“两边都活动者退出再进入”成对给出事件。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-zero")
    body = seek(client, sid, 0).json()
    assert _triples(body["events"]) == [
        (0, "exit", "s1"),
        (0, "enter", "s1"),
    ]
    assert [a["id"] for a in body["active"]] == ["s1"]


def test_seek_events_carry_full_fields_marked_at_target():
    """事件沿用 id/speaker/track/time_ms/type 字段，时刻一律为目标时刻。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-fields")
    advance(client, sid, 900)
    body = seek(client, sid, 2000).json()
    enter_s3 = next(e for e in body["events"] if e["id"] == "s3")
    assert enter_s3 == {
        "id": "s3",
        "speaker": "Alice",
        "track": 1,
        "time_ms": 2000,  # 目标时刻，而非 s3 真实的 start_ms=1000
        "type": "enter",
    }


def test_seek_orders_by_session_subtitle_order():
    """同类事件按会话字幕顺序，而非按轨道或 speaker。"""
    client = TestClient(app)
    client.put("/scenes/seek-order", json={
        "track_count": 3,
        "expected_revision": 0,
        "subtitles": [
            {"id": "a", "speaker": "Z", "start_ms": 0, "end_ms": 5000},
            {"id": "b", "speaker": "Y", "start_ms": 0, "end_ms": 5000},
            {"id": "c", "speaker": "X", "start_ms": 0, "end_ms": 5000},
        ],
    })
    sid = client.post("/sessions", json={"scene_id": "seek-order"}).json()[
        "session_id"
    ]
    advance(client, sid, 100)
    # 定位到 200：三条两侧都活动，先三条 exit（会话序 a,b,c）再三 enter
    body = seek(client, sid, 200).json()
    assert _triples(body["events"]) == [
        (200, "exit", "a"),
        (200, "exit", "b"),
        (200, "exit", "c"),
        (200, "enter", "a"),
        (200, "enter", "b"),
        (200, "enter", "c"),
    ]


# ---------------------------------------------------------------------------
# 隐藏集合
# ---------------------------------------------------------------------------


def test_seek_respects_hidden_speakers_on_both_snapshots():
    """定位按当前隐藏集合过滤两端：隐藏者不退出也不进入。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-hidden")
    advance(client, sid, 900)  # s1(Alice)、s2(Bob) 活动
    set_hidden(client, sid, ["Bob"])  # s2 退出（隐藏切换事件，与本测试无关）

    # 当前可见 active 只有 s1；定位到 2000：可见变化只有 s1 退出、s3 进入，
    # s2 虽两侧都活动但被隐藏，不产生任何事件。
    body = seek(client, sid, 2000).json()
    assert _triples(body["events"]) == [
        (2000, "exit", "s1"),
        (2000, "enter", "s3"),
    ]
    assert [a["id"] for a in body["active"]] == ["s3"]

    # 隐藏集合不被定位改变：清除隐藏后 s2 在 2000 仍活动
    body = set_hidden(client, sid, []).json()
    assert [e["id"] for e in body["events"]] == ["s2"]
    assert [a["id"] for a in body["active"]] == ["s2", "s3"]


def test_seek_does_not_change_snapshot_or_hidden_set():
    """定位不改快照（state/finish 完整字段）与隐藏集合。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-unchanged")
    set_hidden(client, sid, ["Bob"])
    advance(client, sid, 2000)
    seek(client, sid, 100)

    body = client.post(f"/sessions/{sid}/finish").json()
    assert body["cursor_ms"] == 100
    assert [s["id"] for s in body["subtitles"]] == ["s1", "s2", "s3"]
    assert body["assignments"] == [
        {"id": "s1", "track": 1},
        {"id": "s2", "track": 2},
        {"id": "s3", "track": 1},
    ]


# ---------------------------------------------------------------------------
# 补取通道：seq 接续、确认边界保留、倒带按 seq 补取
# ---------------------------------------------------------------------------


def test_seek_channel_events_continue_seq_and_keep_queue():
    """已开通道：未确认队列与确认边界保留，定位事件接续 seq。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-chan")
    open_channel(client, sid)
    advance(client, sid, 1000)  # 入队 s2 enter(800)、s1 exit(1000)、s3 enter(1000)

    # 首次纯读取（ack 0）取得 3 条，再确认前两条，剩 seq=3 未确认
    assert [e["seq"] for e in fetch(client, sid, 0).json()["events"]] == [1, 2, 3]
    body = fetch(client, sid, 2).json()
    assert body["acked_seq"] == 2
    assert [e["seq"] for e in body["events"]] == [3]

    # 定位到 3000：s2、s3（当前可见 active）退出；无 enter
    body = seek(client, sid, 3000).json()
    assert _triples(body["events"]) == [
        (3000, "exit", "s2"),
        (3000, "exit", "s3"),
    ]

    # 通道：旧的未确认 seq=3 原样保留，新事件接续 seq 4、5
    body = fetch(client, sid, 2).json()
    assert body["acked_seq"] == 2
    assert [(e["seq"], e["time_ms"], e["type"], e["id"]) for e in body["events"]] == [
        (3, 1000, "enter", "s3"),
        (4, 3000, "exit", "s2"),
        (5, 3000, "exit", "s3"),
    ]


def test_seek_backward_channel_fetch_by_seq_not_time():
    """倒带后通道不按时间去重：新事件照常接续 seq，补取按 seq。"""
    client = TestClient(app)
    sid = make_session(client, "scene-seek-chanback")
    open_channel(client, sid)
    advance(client, sid, 2000)
    # 先读取使 max_returned_seq 推进，再确认全部 4 条
    # （s2 enter@800、s1 exit@1000、s3 enter@1000、s3 exit@2500 不在窗口；
    # 窗口 (0,2000] 实际 3 条：s2 enter、s1 exit、s3 enter）
    assert [e["seq"] for e in fetch(client, sid, 0).json()["events"]] == [1, 2, 3]
    fetch(client, sid, 3)
    assert fetch(client, sid, 3).json()["events"] == []

    # 倒带回 0：s2、s3 退出，s1 进入（seq 4、5、6）
    seek(client, sid, 0)
    body = fetch(client, sid, 3).json()
    assert body["acked_seq"] == 3  # 确认边界不回退
    assert [(e["seq"], e["time_ms"], e["type"], e["id"]) for e in body["events"]] == [
        (4, 0, "exit", "s2"),
        (5, 0, "exit", "s3"),
        (6, 0, "enter", "s1"),
    ]


def test_seek_channel_full_503_changes_nothing(monkeypatch):
    """容量不足：游标、队列、seq、计时段、播放状态均不变。"""
    monkeypatch.setenv("CHANNEL_EVENT_LIMIT", "3")
    main_module.reset_playback()
    client = TestClient(app)
    sid = make_session(client, "scene-seek-full")
    open_channel(client, sid)
    advance(client, sid, 2000)  # 入队 3 条：容量满

    # 定位需要 4 条（s2/s3 exit + s2/s3 enter，同刻定位到 2000）> 0 剩余
    resp = seek(client, sid, 2000)
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail["requested"] == 4
    assert detail["available"] == 0

    # 游标仍是 2000，队列仍是原来的 3 条，next_seq 未消费
    st = autoplay(client, sid, "status").json()
    assert st == {"session_id": sid, "status": "idle", "cursor_ms": 2000}
    body = fetch(client, sid, 0).json()
    assert [e["seq"] for e in body["events"]] == [1, 2, 3]
    assert main_module.get_playback().channel_capacity.used == 3


def test_seek_with_zero_events_succeeds_when_capacity_full(monkeypatch):
    """满额时无事件的定位仍可执行（不占名额）。"""
    monkeypatch.setenv("CHANNEL_EVENT_LIMIT", "3")
    main_module.reset_playback()
    client = TestClient(app)
    # 场景只有一条字幕，定位到其区间之外：旧 active 在游标 0 时为 s1，
    # 先构造满额：开通道、推进产生 1 条事件，再用别的会话占满剩余 2 条。
    sid = make_session(client, "scene-seek-zeroevt", subtitles=[
        {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
    ])
    open_channel(client, sid)
    advance(client, sid, 1000)  # 1 条 exit 入队
    sid2 = make_session(client, "scene-seek-zeroevt-2", subtitles=[
        {"id": "z", "speaker": "Z", "start_ms": 0, "end_ms": 1000},
        {"id": "y", "speaker": "Y", "start_ms": 0, "end_ms": 50},
    ])
    open_channel(client, sid2)
    advance(client, sid2, 1000)  # 2 条事件（y 的 enter/exit 在 0 已活动？）
    # y[0,50) 在 0 已活动：推进到 1000 的窗口 (0,1000] 产生 y exit 与
    # z exit，共 2 条；合计 3 条，容量满
    assert main_module.get_playback().channel_capacity.used == 3

    # sid 已在 1000（无 active），再定位到 5000：旧/目标均无 active -> 0 事件
    resp = seek(client, sid, 5000)
    assert resp.status_code == 200
    assert resp.json()["events"] == []
    assert resp.json()["cursor_ms"] == 5000


def test_seek_without_channel_does_not_consume_capacity(monkeypatch):
    """未开通道的定位不占共享容量。"""
    monkeypatch.setenv("CHANNEL_EVENT_LIMIT", "3")
    main_module.reset_playback()
    client = TestClient(app)
    sid_a = make_session(client, "scene-seek-nc-a", subtitles=[
        {"id": "a", "speaker": "A", "start_ms": 0, "end_ms": 1000},
    ])
    open_channel(client, sid_a)
    advance(client, sid_a, 1000)  # 1 条 exit@1000 入队
    # 另一个会话再占 2 条（b 在窗口内成对 enter/exit）
    sid_fill = make_session(client, "scene-seek-nc-fill", subtitles=[
        {"id": "b", "speaker": "B", "start_ms": 100, "end_ms": 200},
    ])
    open_channel(client, sid_fill)
    advance(client, sid_fill, 1000)
    assert main_module.get_playback().channel_capacity.remaining == 0

    sid_b = make_session(client, "scene-seek-nc-b")
    # 未开通道：定位照常成功（s1 在 0 已活动，只产生其 3000 时刻的退出；
    # 即使有显示事件也不占共享容量）
    resp = seek(client, sid_b, 3000)
    assert resp.status_code == 200
    assert len(resp.json()["events"]) == 1


# ---------------------------------------------------------------------------
# 自动播放：状态迁移、旧驱动不推进、恢复重新计时
# ---------------------------------------------------------------------------


def test_seek_idle_session_stays_idle():
    client = TestClient(app)
    sid = make_session(client, "scene-seek-ap-idle")
    body = seek(client, sid, 1000).json()
    assert body["status"] == "idle"
    assert autoplay(client, sid, "status").json()["status"] == "idle"


def test_seek_running_session_pauses_and_old_driver_segment_dead():
    """运行中定位：状态 paused；旧计时段的后续 tick 不得推进游标。"""
    clock = _FakeClock(0)
    pm = PlaybackManager(
        channel_event_limit=64,
        autoplay_tick_interval_s=None,
        clock=clock,
    )
    session = _open_record_session(pm)
    pm.start_autoplay(session, now=clock())
    clock.advance_ms(200)
    pm.tick_autoplay(now=clock())
    assert session.cursor_ms == 200

    # 定位到 2000（在锁内完成 paused 迁移）
    state = pm.seek(session, 2000)
    assert state["status"] == AUTOPLAY_PAUSED
    assert session.cursor_ms == 2000

    # 旧驱动段（base=0, base_monotonic=0）即使再经过很久也不得推进：
    # tick 看到 paused 直接返回，游标停在 2000
    clock.advance_ms(5000)
    pm.tick_autoplay(now=clock())
    assert session.autoplay_status == AUTOPLAY_PAUSED
    assert session.cursor_ms == 2000

    # resume 从新游标 2000 开新计时段，不追赶
    pm.resume_autoplay(session, now=clock())
    clock.advance_ms(100)
    pm.tick_autoplay(now=clock())
    assert session.cursor_ms == 2100


def test_seek_paused_session_remains_paused_with_new_cursor():
    client = TestClient(app)
    sid = make_session(client, "scene-seek-ap-paused")
    open_channel(client, sid)
    autoplay(client, sid, "start")
    autoplay(client, sid, "pause")
    body = seek(client, sid, 1500).json()
    assert body["status"] == "paused"
    assert autoplay(client, sid, "status").json() == {
        "session_id": sid,
        "status": "paused",
        "cursor_ms": 1500,
    }
    # resume 从 1500 起正常恢复
    assert autoplay(client, sid, "resume").json()["status"] == "running"


def test_seek_blocked_session_pauses_and_503_leaves_blocked():
    """blocked 定位成功 -> paused；容量不足 503 时保持 blocked 与原游标。"""
    clock = _FakeClock(0)
    pm = PlaybackManager(
        channel_event_limit=3,
        autoplay_tick_interval_s=None,
        clock=clock,
    )
    # 会话 B 先占满全部 3 个名额（0 已活动的字幕不产生事件：窗口
    # (0,1000] 中 b[100,400) 成对 enter/exit，c[200,600) 成对 -> 4 条太多；
    # 选只产生 3 条的场景）
    filler = _open_record_session(
        pm,
        subtitles=[
            {"id": "a", "speaker": "A", "start_ms": 0, "end_ms": 100},
            {"id": "b", "speaker": "B", "start_ms": 100, "end_ms": 200},
        ],
        tracks=[1, 2],
    )
    # 窗口 (0,300]：b enter@100、a exit@100、b exit@200 = 3 条
    pm.advance(filler, 300)
    assert pm.channel_capacity.remaining == 0

    # 会话 A：后台推进到 2000 需要 3 条 > 剩余 0 -> blocked，游标不动
    session = _open_record_session(pm)
    pm.start_autoplay(session, now=clock())
    clock.advance_ms(2000)
    pm.tick_autoplay(now=clock())
    assert session.autoplay_status == AUTOPLAY_BLOCKED
    assert session.cursor_ms == 0

    # 容量仍满：定位 0 -> 2000 需要 3 条（s1 exit + s2/s3 enter）-> 503，
    # 状态仍 blocked、游标、队列不变
    with pytest.raises(Exception):
        pm.seek(session, 2000)
    assert session.autoplay_status == AUTOPLAY_BLOCKED
    assert session.cursor_ms == 0

    # 确认释放 B 的 3 条后定位成功 -> paused、游标 2000
    pm.fetch_channel(filler, 0)  # 先纯读取推进 max_returned_seq
    pm.fetch_channel(filler, 3)
    assert pm.channel_capacity.remaining == 3
    state = pm.seek(session, 2000)
    assert state["status"] == AUTOPLAY_PAUSED
    assert session.cursor_ms == 2000


# ---------------------------------------------------------------------------
# HTTP 校验与 404
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_time", [True, False, "100", "0", 1.5, -1, -0.1, None])
def test_seek_rejects_bool_string_float_negative_422(bad_time):
    client = TestClient(app)
    sid = make_session(client, f"scene-seek-bad-{str(bad_time)!r}")
    resp = client.post(f"/sessions/{sid}/seek", json={"time_ms": bad_time})
    assert resp.status_code == 422
    # 非法请求不改游标
    assert seek(client, sid, 0).json()["cursor_ms"] == 0


def test_seek_missing_field_422():
    client = TestClient(app)
    sid = make_session(client, "scene-seek-missing")
    assert client.post(f"/sessions/{sid}/seek", json={}).status_code == 422


def test_seek_unknown_session_404():
    client = TestClient(app)
    assert seek(client, "nope", 100).status_code == 404


def test_seek_finished_session_404():
    client = TestClient(app)
    sid = make_session(client, "scene-seek-finished")
    client.post(f"/sessions/{sid}/finish")
    assert seek(client, sid, 100).status_code == 404


def test_seek_response_shape_and_order():
    client = TestClient(app)
    sid = make_session(client, "scene-seek-shape")
    advance(client, sid, 900)
    resp = seek(client, sid, 2000)
    assert set(resp.json().keys()) == {
        "session_id",
        "scene_id",
        "source_revision",
        "cursor_ms",
        "events",
        "active",
        "status",
    }
    # 旧推进接口响应保持原样（无 events/status 字段）
    body = advance(client, sid, 2100).json()
    assert "events" not in body and "status" not in body


# ---------------------------------------------------------------------------
# 确定性管理器测试辅助
# ---------------------------------------------------------------------------


class _FakeClock:
    """可控单调时钟：读数为十进制精确的秒值（Decimal），与生产 time.monotonic
    一样为 float 之外的数值类型——驱动用 int((t1-t0)*1000) 取整毫秒，
    Decimal 差值精确，避免二进制浮点在 5 秒附近 100ms 差被截成 99ms。"""

    def __init__(self, now_ms=0):
        from decimal import Decimal

        self._decimal = Decimal
        self._now_ms = now_ms

    def __call__(self):
        return self._decimal(self._now_ms) / self._decimal(1000)

    def advance_ms(self, ms):
        self._now_ms += ms


def _open_record_session(
    pm, subtitles=None, tracks=None, scene_id="scene-unit-seek", track_count=2
):
    if subtitles is None:
        # s1[0,1000) s2[800,3000) s3[1000,2500)：分轨 1/2/1
        subtitles = [
            {"id": "s1", "speaker": "Alice", "start_ms": 0, "end_ms": 1000},
            {"id": "s2", "speaker": "Bob", "start_ms": 800, "end_ms": 3000},
            {"id": "s3", "speaker": "Alice", "start_ms": 1000, "end_ms": 2500},
        ]
        tracks = [1, 2, 1]
    elif tracks is None:
        tracks = list(range(1, len(subtitles) + 1))
    record = SceneRecord(
        scene_id=scene_id,
        revision=1,
        track_count=track_count,
        subtitles=subtitles,
        assignments=[
            {"id": s["id"], "track": t} for s, t in zip(subtitles, tracks)
        ],
        changes=0,
    )
    state = pm.create(record)
    session = pm.get(state["session_id"])
    pm.open_channel(session)
    return session
