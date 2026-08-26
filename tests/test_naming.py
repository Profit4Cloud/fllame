from fllame.recipes.naming import derive_handle


def test_takes_part_after_last_slash():
    assert derive_handle("meta-llama/Meta-Llama-3-8B-Instruct") == "meta-llama-3-8b-instruct"


def test_lowercases_and_slugifies():
    assert derive_handle("org/Some_Model.Name v2") == "some-model-name-v2"


def test_no_slash_uses_whole_repo_id():
    assert derive_handle("standalone-name") == "standalone-name"


def test_falls_back_to_model_when_nothing_left():
    assert derive_handle("org/___") == "model"
