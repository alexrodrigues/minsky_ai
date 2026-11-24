# Minsky Setup Prompt

Copy the entire content of this file into your coding agent (Claude, Codex, etc.) so it can perform the setup steps automatically for you.

1. **Read the constraints**
   a. Open and read `constraints/architecture.md` in its entirety.
   b. Open and read `constraints/config-schema.md` in its entirety.
   c. Open and read `constraints/non-goals.md` in its entirety.
   d. Open and read `constraints/setup-contract.md` in its entirety.
   Only after all four files are fully read may you proceed.

2. **Verify Ollama**
   a. Run `ollama --version`.
   b. Check that something is listening on `localhost:11434`.
   c. If Ollama is not installed or not running, output a clear message telling the user to install it (e.g., `brew install ollama` on macOS) and terminate the setup.

3. **Verify required models**
   a. Run `ollama list` to list all models currently available locally.
   b. Check that the embedding model `nomic-embed-text` is present.
   c. Check that at least one worker model is present; recommend `gpt-oss:20b` or `qwen2.5-coder:14b`.
   d. Note that each of these worker models requires **at least 13 GB of free RAM** to run comfortably.
   e. If the user is unsure whether their machine has that much free RAM, suggest `qwen2.5-coder:7b` as a lighter fallback.
   f. For any model that is missing, inform the user of the exact model name, give an approximate download size (e.g., `gpt-oss:20b` ~ 13 GB, `qwen2.5-coder:14b` ~ 9 GB, `qwen2.5-coder:7b` ~ 5 GB - these are approximate, the exact size shows during the pull), and ask for explicit confirmation before pulling.
   g. If the user confirms, run `ollama pull <model-name>` for each missing model. If the pull fails, terminate the setup and report the error.

4. **Ask which folders to index**
   a. Prompt the user for the paths of the directories they want to index (e.g., `~/Developer`, `~/Projects`).
   b. Do not default to indexing the entire home directory without explicit consent, as required by `constraints/setup-contract.md`.
   c. Store the user's answer(s) for use in step 5.

5. **Edit `minsky.toml`**
   a. Open `minsky.toml`.
   b. Under the `[index]` section, set the `roots` key to the list of directories obtained in step 4.
   c. If the user indicated any projects that live outside those roots, add them to `extra_projects`.
   d. Do **not** add, remove, or rename any keys beyond what is defined in `constraints/config-schema.md`.
   e. Preserve all other existing keys and values exactly as they appear in the file.

6. **Create a virtual environment and install the package**
   a. Run `python3 -m venv .venv`.
   b. Activate it with `source .venv/bin/activate`, or skip activation and call `.venv/bin/pip`/`.venv/bin/minsky` directly (the convention the rest of these steps use).
   c. Install the package in editable mode: `.venv/bin/pip install -e .`.
   d. If any command fails (e.g., `pip install` error), stop immediately and report the error.

7. **Run the index**
   a. Execute `.venv/bin/minsky index`.
   b. Capture its output and report back what directories and files were indexed, including the number of projects, number of chunks, and any warnings or skipped files.

8. **Suggest a first real query**
   a. Recommend a simple search command that refers to the user's own code, e.g.:
      ```bash
      .venv/bin/minsky search "refactor login flow" --project myapp
      ```
   b. Explain that the `--project` flag can narrow the search to a specific project if desired.
   c. Mention that `.venv/bin/minsky eval` is available for measuring retrieval quality with an `evalset.toml` of the user's own test cases, and that the shipped `evalset.toml` tests the minsky source itself.

9. **Final reminders**
   a. Reiterate that no Git operations (commit, push, or repository creation) may occur during this setup, per `constraints/setup-contract.md`.
   b. If any step fails (Ollama not installed, model pull failure, virtual-environment creation failure, index failure), stop immediately and output a clear error message. Do not proceed past a broken setup.