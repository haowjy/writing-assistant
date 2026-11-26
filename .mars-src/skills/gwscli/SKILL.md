---
name: gwscli
version: 1.0.0
description: >-
  Provides top-level guidance for the Google Workspace CLI (gwscli), focusing on
  authentication, safety in headless environments, and general utilities. Use when
  the user needs to log in, verify the CLI version, or handle general execution safety.
category: workspace_productivity
metadata:
  display_name: Google Workspace CLI
  safety_profile: agent-safe
  min_cli_version: 0.2.0
---

# Google Workspace CLI (gwscli) Top-Level User Guide

This guide describes how to initialize and run the Google Workspace CLI
(`gwscli`) and use its general utility commands.

--------------------------------------------------------------------------------

## 1. Setup & Scopes (Authentication)

`gwscli` requires authentication to access your Google Workspace APIs.
Authentication uses incremental OAuth scopes, allowing you to authorize single
commands, multiple commands, entire applications, or all workspace services at
once.

### Custom GCP Project Setup

You can either configure your own GCP OAuth 2.0 client in the GWSCLI and then
run auth login commands (Built-in-Auth), or you can manage OAuth tokens yourself
and pass them to the CLI (Self-Managed-Auth).

*   **Built-in-Auth**: Configure your own GCP OAuth 2.0 client JSON file
    downloaded from the Google Cloud Console (see
    [Create OAuth client ID credentials](https://developers.google.com/workspace/guides/create-credentials#oauth-client-id)):

    ```bash
    gwscli auth setup --oauth_client_file=/path/to/client_secret.json
    ```

    Run `gwscli auth login` to authenticate and authorize access.

*   **Self-Managed-Auth**: Pass an explicit OAuth access token directly on any
    command:

    ```bash
    gwscli <service> <action> --oauth_token="ya29..."
    ```

### Interactive Login

The CLI prompts you to open an authorization URL in your browser, log in with
your Google Workspace account, and grant the required permissions.

```bash
gwscli auth login [options] [targets...]
```

#### Authentication Options & Flavours:

1.  **All Applications & Commands (Full Workspace Auth)**:

    ```bash
    gwscli auth login
    ```

    *Generates an authorization URL with scopes across all available Google
    Workspace tools.*

2.  **Entire Application(s)**:

    ```bash
    # Single application
    gwscli auth login chat

    # Multiple applications
    gwscli auth login chat gmail
    ```

3.  **Multiple Specific Commands**:

    ```bash
    gwscli auth login "chat search_messages" "chat send_message" "gmail search_threads"
    ```

4.  **Mix of Applications & Commands**:

    ```bash
    gwscli auth login "gmail" "chat search_messages"
    ```

5.  **Single Command (Unquoted / Legacy)**:

    ```bash
    gwscli auth login drive list_recent_files
    gwscli auth login gmail search_threads
    ```

6.  **Read-Only Bundle (`--bundle=readonly`)**:

    ```bash
    # Read-only for all workspace applications
    gwscli auth login --bundle=readonly

    # Read-only for a specific application
    gwscli auth login chat --bundle=readonly
    ```

### Manual Token Refresh, Logout, & Reset

*   **Refresh Existing Authorized Scopes**:

    ```bash
    gwscli auth refresh
    ```

    *(Note: `gwscli` automatically refreshes expired tokens in the background
    when executing commands. Run this only if you want to manually force a token
    refresh.)*

*   **Log Out & Revoke Credentials**:

    ```bash
    gwscli auth logout
    ```

*   **Reset Authentication & Remove Custom OAuth Client Credentials**:

    ```bash
    gwscli auth reset
    ```

    *(Note: This logs out the current user and removes any custom OAuth client
    file specified during `gwscli auth setup`. You will need to re-run `gwscli
    auth setup` to authenticate again.)*

*   **SSH / Remote Machine Note**: The OAuth flow runs a temporary server on
    `localhost` to listen for the redirect callback. Therefore, the OAuth link
    must be opened on the same machine that executed the `gwscli auth login`
    command. If you are logged into a remote machine via SSH, you should run the
    login flow locally, or set up SSH port forwarding back to your local machine
    for the dynamic port printed by the command.

--------------------------------------------------------------------------------

## 2. Headless Safety & Execution Guidelines

When executing in a headless or non-interactive environment, you **must**
strictly adhere to the following safety guidelines to prevent unresponsive
states and execution failures.

### A. Prevent Interactive Deadlocks in Scripts (Human Hand-Off)

*   **CRITICAL**: You **must never** use `gwscli auth login` inside a script,
    background process, or automated workflow. This command is strictly
    interactive and requires a browser-based authentication flow that cannot
    complete in headless or non-interactive contexts. Running it in a script or
    automated workflow will block the execution indefinitely, causing a timeout
    deadlock.
*   **Action**: If any command fails with an authentication error (e.g., `failed
    to obtain token` or `Run gwscli auth login first`), you **must** immediately
    pause execution and request the human user to perform the login manually in
    their own local terminal:

    > *"I need to access Google Workspace, but I am currently unauthenticated.
    > Please run `gwscli auth login <command>` (e.g., `gwscli auth login drive
    > search_files`) in your local terminal to authorize my access, and let me
    > know when you are done."*

### B. Interactive Human-in-the-Loop (HITL) Confirmation (Destructive Operations)

*   **Feature**: The CLI is programmed to intercept destructive requests and
    prompt the user before proceeding.
*   **Action for AI Agents**: If a command hits a Human-in-the-Loop (HITL)
    interception, the CLI process will pause and print a prompt to `STDERR`
    asking for STDIN confirmation. If you are an AI assistant executing this
    command in the background, you must monitor the process output. When you see
    the prompt, you MUST ask the human user for their explicit permission
    directly. Once the user replies, you must pipe their choice (e.g.,
    `"ALLOW_ONCE\n"` or `"DENY_ONCE\n"`) to the command's `STDIN` buffer to
    proceed (using whichever process management mechanism your platform
    provides, such as sending input to a background task). Persistent `*_ALWAYS`
    approval options require an interactive terminal (TTY) and are disabled when
    confirming via piped or non-TTY `STDIN`.

### C. Parsing Command Output (JSON vs Markdown vs Plain Text)

*   **Output Behavior**: `gwscli` writes command output directly to `stdout`.
    The output format varies by command:
    1.  **JSON Objects & Lists**: Most resource mutation and query tools (e.g.
        `drive search_files`, `calendar list_events`, `chat send_message`, `docs
        read_doc`, `slides read_presentation`) output structured JSON.
    2.  **Plain Text / Markdown**: Document reading tools (e.g. `drive
        read_file_content`) output direct Markdown/text strings.
    3.  **Plain Text Utility Output**: Static utility commands (e.g. `version`,
        `skills list`, `auth`) output plain text strings.
*   **Inspecting Output Format (`--help`) & Detailed Structure (`--schema`)**:
    *   Run `gwscli <app> <tool> --help` to inspect the high-level **Output
        Format** summary.
    *   Run `gwscli <app> <tool> --schema` to print the full JSON schema /
        response structure and field definitions without executing a backend
        call.
*   **Output Configuration Flags**: Some commands provide flags that customize
    the response payload (such as `--excludeContentSnippets` or export format
    options). Consult `gwscli <app> <tool> --help` for available options.
*   **Parsing JSON Tools with `jq`**: For tools returning structured JSON, pipe
    stdout to `jq`:

    *   *Example:*

        ```bash
        gwscli drive search_files --query="title contains 'mydoc'" | jq '.files[] | {id: .id, url: .viewUrl}'
        ```

*   **Handling Markdown / Plain Text Tools**: Do **not** pipe document content
    tools (e.g. `drive read_file_content`) to `jq`, as raw Markdown will cause
    JSON parsing errors. Read stdout directly as text.

### D. How to Operationalize Skill Metadata

Parse and verify the YAML metadata properties declared at the top of this file
during your execution initialization:

1.  **`min_cli_version: 0.2.0`**: Run `gwscli version` before executing any
    commands to verify that the compiled CLI binary version is >= `0.2.0`.
2.  **`safety_profile: agent-safe`**: This profile indicates that the CLI
    executes API-scoped operations. However, you must still prompt the human
    user for explicit confirmation before executing any write or destructive
    operation.

--------------------------------------------------------------------------------

## 3. Running the CLI

Run the compiled binary directly from your command line:

```bash
gwscli <command> [flags]
```

To see available commands or get help for a specific command, use the `--help`
flag:

```bash
gwscli --help
```

*(Note: Running `--help` on domain-specific commands dynamically queries the
backend to return rich, up-to-date parameter details).*

--------------------------------------------------------------------------------

## 4. General Utility Commands

The following general utilities are supported natively by the top-level CLI and
do not require authentication:

### A. Print version

```bash
gwscli version
```

--------------------------------------------------------------------------------

## 5. Managing Agent Skills

The `gwscli` tool provides agent skills that can be installed and used by AI
agents.

### A. List Available Skills

To list all available agent skills in the CLI:

```bash
gwscli skills list
```

### B. Install Skills

*   **Requirement**: You **must** ask for explicit confirmation from the human
    user before installing skills to verify the intended installation path.
*   **Best Practice**: Recommend installing skills to a standard directory where
    agents automatically discover and read skills from, such as
    `~/.agents/skills`.

To install these skills to a target directory:

```bash
gwscli skills install --dir <target_directory>
```

Options:

*   `--dir <directory>`: (Required) The target directory where skills should be
    installed. Supports tilde expansion (e.g., `~/.agents/skills`).
*   `--overwrite`: (Optional) Overwrite existing skills in the destination
    directory if they already exist.

*Example:*

```bash
gwscli skills install --dir ~/.agents/skills
```

--------------------------------------------------------------------------------

## 6. Troubleshooting

### Authentication Errors

*   **Error**: `failed to obtain token source: ... Run gwscli auth login
    first...`
*   **Resolution**: Request the user to run `gwscli auth login <command>` in
    their terminal to authenticate and refresh access credentials for that
    domain.
