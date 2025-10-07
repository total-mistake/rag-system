#!/usr/bin/env python3
"""
Скрипт для запуска интеграционных тестов индексатора
"""
import subprocess
import sys
import argparse
import time


def run_command(cmd):
    """Выполняет команду и выводит результат"""
    print(f"Выполняем: {' '.join(cmd)}")
    print("-" * 50)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    print("-" * 50)
    print(f"Код возврата: {result.returncode}")
    print(f"Время выполнения: {end_time - start_time:.2f} секунд")
    print()
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Запуск интеграционных тестов индексатора")
    parser.add_argument("--all", action="store_true", help="Запустить все тесты")
    parser.add_argument("--integration", action="store_true", help="Запустить интеграционные тесты")
    parser.add_argument("--search", action="store_true", help="Запустить только тесты векторного поиска")
    parser.add_argument("--coverage", action="store_true", help="Запустить с покрытием кода")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод")
    parser.add_argument("--test", help="Запустить конкретный тест")
    parser.add_argument("--fast", action="store_true", help="Быстрые тесты (пропустить медленные)")
    
    args = parser.parse_args()
    
    # Базовая команда pytest
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.extend(["-v", "-s"])  # -s для вывода print'ов
    
    # Выбор тестов
    if args.search:
        cmd.extend(["-m", "search"])
        print("🔍 Запускаем тесты ВЕКТОРНОГО ПОИСКА...")
    elif args.integration:
        cmd.extend(["-m", "integration"])
        print("🔗 Запускаем ИНТЕГРАЦИОННЫЕ тесты...")
    elif args.test:
        cmd.extend(["-k", args.test])
        print(f"🎯 Запускаем конкретный тест: {args.test}")
    else:
        print("🚀 Запускаем ВСЕ интеграционные тесты...")
        cmd.append("tests/test_indexer_integration.py")
    
    # Пропуск медленных тестов
    if args.fast:
        cmd.extend(["-m", "not slow"])
        print("⚡ Пропускаем медленные тесты")
    
    # Покрытие кода
    if args.coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
        print("📊 С анализом покрытия кода")
    
    print()
    print("⚠️  ВНИМАНИЕ: Тесты используют реальные модели эмбеддингов и ChromaDB")
    print("⏱️  Первый запуск может занять время на загрузку модели")
    print("💾 Тестовая БД создается во временной директории")
    print()
    
    success = run_command(cmd)
    
    if success:
        print("✅ Все тесты прошли успешно!")
        print("🎉 Векторный поиск работает корректно!")
        if args.coverage:
            print("📊 Отчет о покрытии сохранен в htmlcov/index.html")
    else:
        print("❌ Некоторые тесты не прошли!")
        print("🔧 Проверьте логи выше для диагностики проблем")
        sys.exit(1)


if __name__ == "__main__":
    # Примеры использования
    if len(sys.argv) == 1:
        print("🧪 Скрипт для запуска интеграционных тестов индексатора")
        print()
        print("⚠️  ВАЖНО: Эти тесты используют реальные модели и БД!")
        print()
        print("Примеры использования:")
        print("  python run_tests.py --all              # Все интеграционные тесты")
        print("  python run_tests.py --integration      # Тесты индексации")
        print("  python run_tests.py --search           # Только тесты поиска")
        print("  python run_tests.py --fast             # Быстрые тесты (без медленных)")
        print("  python run_tests.py --coverage         # С покрытием кода")
        print("  python run_tests.py --test vector      # Конкретный тест")
        print("  python run_tests.py -v --search        # Подробный вывод тестов поиска")
        print()
        print("Запускаем все тесты по умолчанию...")
        sys.argv.append("--all")
    
    main()