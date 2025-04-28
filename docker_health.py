#!/usr/bin/env python3
import time
import psutil
import subprocess
import sys

def is_port_listening(port: int) -> bool:
    """
    Возвращает True, если в списке активных соединений (inet)
    найдено соединение с локальным портом port в статусе LISTEN.
    """
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr and conn.laddr.port == port and conn.status == 'LISTEN':
            return True
    return False

def restart_docker():
    """
    Выполняет последовательный запуск команд для перезапуска docker-compose:
      1. docker compose -p digital-assistant-first-aviasales-new-latest down
      2. docker compose -p digital-assistant-first-aviasales-new-latest up -d
      3. docker system prune -a --volumes -f
    Все команды выполняются в каталоге проекта.
    """
    project_dir = "/root/PRODUCTION/Digital-Assistant-First-New-Aviasales-Latest"
    commands = [
        "docker compose -p digital-assistant-first-aviasales-new-latest down",
        "docker compose -p digital-assistant-first-aviasales-new-latest up -d",
        "docker system prune -a --volumes -f"
    ]
    
    for cmd in commands:
        try:
            print(f"Выполняется команда: {cmd}")
            result = subprocess.run(cmd, shell=True, cwd=project_dir,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(10)
            if result.returncode != 0:
                print(f"Ошибка выполнения команды: {cmd}")
                print(f"Ошибка: {result.stderr}")
                # При ошибке можно завершить выполнение или продолжить, если это допустимо
                return False
            else:
                print(result.stdout)
        except Exception as e:
            print(f"Исключение при выполнении команды {cmd}: {e}")
            return False
    return True

def main():
    port = 9777
    while True:
        print("Проверка порта...")
        if is_port_listening(port):
            print(f"Порт {port} прослушивается. Всё в порядке.")
        else:
            print(f"Порт {port} не обнаружен! Перезапуск Docker-контейнеров...")
            restart_docker()
        # Ждем 60 секунд до следующей проверки
        time.sleep(60)

if __name__ == '__main__':
    main()