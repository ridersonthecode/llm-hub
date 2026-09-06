"""CLI zur Verwaltung von users.json (Website-Login fürs Dashboard, siehe
web_auth.py) - bewusst kein Web-UI dafür: Nutzer anlegen/löschen ist ein
seltener, administrativer Vorgang, den man genauso gut vom Terminal aus
erledigt, ohne dafür eine eigene Signup-/Verwaltungsseite mit eigener
Angriffsfläche zu bauen.

Nutzung:
    python -m llm_hub.manage_users add <username>      # legt an oder ändert Passwort
    python -m llm_hub.manage_users remove <username>
    python -m llm_hub.manage_users list

Solange users.json fehlt oder leer ist, bleibt die Website ungeschützt
(siehe web_auth.py) - der erste 'add'-Aufruf aktiviert den Login-Zwang."""
from __future__ import annotations

import argparse
import getpass
import sys

from .web_auth import USERS_PATH, hash_password, read_users, write_users


def _prompt_password() -> str:
    while True:
        pw = getpass.getpass("Passwort: ")
        if len(pw) < 4:
            print("Passwort zu kurz (mind. 4 Zeichen).", file=sys.stderr)
            continue
        if getpass.getpass("Passwort wiederholen: ") != pw:
            print("Passwörter stimmen nicht überein, bitte erneut.", file=sys.stderr)
            continue
        return pw


def cmd_add(username: str) -> None:
    users = read_users()
    action = "geändert" if username in users else "angelegt"
    password = _prompt_password()
    salt_hex, hash_hex = hash_password(password)
    users[username] = {"salt": salt_hex, "hash": hash_hex}
    write_users(users)
    print(f"Nutzer '{username}' {action} ({USERS_PATH}).")


def cmd_remove(username: str) -> None:
    users = read_users()
    if username not in users:
        print(f"Nutzer '{username}' existiert nicht.", file=sys.stderr)
        sys.exit(1)
    del users[username]
    write_users(users)
    if users:
        print(f"Nutzer '{username}' entfernt.")
    else:
        print(f"Nutzer '{username}' entfernt. users.json ist jetzt leer - die Website ist ab sofort ungeschützt.")


def cmd_list() -> None:
    users = read_users()
    if not users:
        print("Keine Nutzer konfiguriert - die Website ist aktuell ungeschützt.")
        return
    for username in sorted(users):
        print(username)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verwaltet users.json fürs Dashboard-Login.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Nutzer anlegen oder Passwort ändern")
    p_add.add_argument("username")

    p_remove = sub.add_parser("remove", help="Nutzer entfernen")
    p_remove.add_argument("username")

    sub.add_parser("list", help="Konfigurierte Nutzer auflisten")

    args = parser.parse_args()
    if args.command == "add":
        cmd_add(args.username)
    elif args.command == "remove":
        cmd_remove(args.username)
    elif args.command == "list":
        cmd_list()


if __name__ == "__main__":
    main()
