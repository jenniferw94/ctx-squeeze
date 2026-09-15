import io
import json

from ctx_squeeze.cli import main


def test_document_mode_writes_result_to_output_file(tmp_path):
    src = tmp_path / "notes.md"
    src.write_text("first paragraph\n\nsecond paragraph\n\nthird paragraph")
    out = tmp_path / "out.md"

    code = main(["--budget", "1000", "-o", str(out), str(src)])

    assert code == 0
    assert out.read_text() == "first paragraph\n\nsecond paragraph\n\nthird paragraph\n"


def test_document_mode_prints_to_stdout_by_default(tmp_path, capsys):
    src = tmp_path / "notes.md"
    src.write_text("only paragraph here")

    code = main(["--budget", "1000", str(src)])

    assert code == 0
    assert capsys.readouterr().out == "only paragraph here\n"


def test_document_mode_stats_go_to_stderr(tmp_path, capsys):
    src = tmp_path / "notes.md"
    src.write_text("alpha\n\nbeta\n\ngamma")

    code = main(["--budget", "1000", "--stats", str(src)])

    assert code == 0
    err = capsys.readouterr().err
    assert "kept 3 of 3 segments" in err
    assert "(budget 1000)" in err


def test_document_mode_json_report_has_expected_fields(tmp_path, capsys):
    src = tmp_path / "notes.md"
    src.write_text("alpha\n\nbeta")

    code = main(["--budget", "1000", "--json", str(src)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"] == "alpha\n\nbeta"
    assert payload["segments_in"] == 2
    assert payload["segments_out"] == 2
    assert payload["notes"] == []


def test_document_mode_unknown_strategy_reports_error(tmp_path, capsys):
    src = tmp_path / "notes.md"
    src.write_text("some text")

    code = main(["--budget", "10", "--strategy", "bogus", str(src)])

    assert code == 1
    assert "bogus" in capsys.readouterr().err


def test_reads_stdin_when_input_is_dash(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("stdin paragraph"))

    code = main(["--budget", "1000", "-"])

    assert code == 0
    assert capsys.readouterr().out == "stdin paragraph\n"


def test_messages_mode_default_output_is_json_array_with_marker(tmp_path, capsys):
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "an old unrelated question about widgets"},
        {"role": "assistant", "content": "an old unrelated answer about widgets"},
        {"role": "user", "content": "the current question"},
        {"role": "assistant", "content": "the current answer"},
    ]
    src = tmp_path / "chat.json"
    src.write_text(json.dumps(history))

    code = main(["--messages", "--budget", "1", "--recent-turns", "1", str(src)])

    assert code == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered[0] == {"role": "system", "content": "sys"}
    assert rendered[1] == {"role": "system", "content": "[2 earlier messages elided]"}
    assert rendered[2]["content"] == "the current question"
    assert rendered[3]["content"] == "the current answer"


def test_messages_mode_no_marker_omits_elision_placeholder(tmp_path, capsys):
    history = [
        {"role": "user", "content": "an old unrelated question about widgets"},
        {"role": "assistant", "content": "an old unrelated answer about widgets"},
        {"role": "user", "content": "the current question"},
    ]
    src = tmp_path / "chat.json"
    src.write_text(json.dumps(history))

    code = main(["--messages", "--budget", "1", "--recent-turns", "1", "--no-marker", str(src)])

    assert code == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == [{"role": "user", "content": "the current question"}]


def test_messages_mode_json_report_has_expected_fields(tmp_path, capsys):
    history = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hello there"},
    ]
    src = tmp_path / "chat.json"
    src.write_text(json.dumps(history))

    code = main(["--messages", "--budget", "1000", "--json", str(src)])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["messages_in"] == 2
    assert payload["messages_out"] == 2
    assert len(payload["messages"]) == 2


def test_messages_mode_non_array_input_reports_error(tmp_path, capsys):
    src = tmp_path / "chat.json"
    src.write_text(json.dumps({"not": "an array"}))

    code = main(["--messages", "--budget", "1000", str(src)])

    assert code == 1
    assert "JSON array" in capsys.readouterr().err


def test_messages_mode_invalid_json_reports_error(tmp_path, capsys):
    src = tmp_path / "chat.json"
    src.write_text("not json at all")

    code = main(["--messages", "--budget", "1000", str(src)])

    assert code == 1
    assert "invalid JSON" in capsys.readouterr().err


def test_missing_input_file_reports_error(tmp_path, capsys):
    code = main(["--budget", "1000", str(tmp_path / "missing.md")])

    assert code == 1
    assert capsys.readouterr().err
