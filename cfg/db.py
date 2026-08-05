import sqlite3
from pathlib import Path
from datetime import datetime
from cfg import DATA_DIR

DB_PATH = DATA_DIR / "queue.db"

class TranslationQueueDB:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queue (
                manga_id TEXT NOT NULL,
                chapter_number TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP NULL,
                completed_at TIMESTAMP NULL,
                error_message TEXT NULL,
                PRIMARY KEY (manga_id, chapter_number)
            )
        """)
        self.conn.commit()

    def add_to_queue(self, manga_id: str, chapter_number: str, source_lang: str):
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT INTO queue (manga_id, chapter_number, source_lang) VALUES (?, ?, ?)",
                           (manga_id, chapter_number, source_lang))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Entry already exists
            return False

    def get_pending_tasks(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT manga_id, chapter_number, source_lang FROM queue WHERE status = 'pending' ORDER BY added_at ASC")
        return [{
            "manga_id": row[0],
            "chapter_number": row[1],
            "source_lang": row[2],
        } for row in cursor.fetchall()]

    def get_processing_tasks(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT manga_id, chapter_number, source_lang FROM queue WHERE status = 'processing' ORDER BY started_at ASC")
        return [{
            "manga_id": row[0],
            "chapter_number": row[1],
            "source_lang": row[2],
        } for row in cursor.fetchall()]

    def get_all_tasks(self) -> list[dict]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT manga_id, chapter_number, source_lang, status, added_at, started_at, completed_at, error_message FROM queue ORDER BY added_at ASC")
        return [{
            "manga_id": row[0],
            "chapter_number": row[1],
            "source_lang": row[2],
            "status": row[3],
            "added_at": row[4],
            "started_at": row[5],
            "completed_at": row[6],
            "error_message": row[7],
        } for row in cursor.fetchall()]

    def get_pending_count(self, manga_id: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM queue WHERE manga_id = ? AND status = 'pending'", (manga_id,))
        return cursor.fetchone()[0]

    def get_processing_count(self, manga_id: str) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM queue WHERE manga_id = ? AND status = 'processing'", (manga_id,))
        return cursor.fetchone()[0]

    def update_task_status(self, manga_id: str, chapter_number: str, status: str, error_message: str | None = None):
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        if status == 'processing':
            cursor.execute("UPDATE queue SET status = ?, started_at = ? WHERE manga_id = ? AND chapter_number = ?",
                           (status, now, manga_id, chapter_number))
        elif status == 'completed':
            cursor.execute("UPDATE queue SET status = ?, completed_at = ? WHERE manga_id = ? AND chapter_number = ?",
                           (status, now, manga_id, chapter_number))
        elif status == 'failed':
            cursor.execute("UPDATE queue SET status = ?, completed_at = ?, error_message = ? WHERE manga_id = ? AND chapter_number = ?",
                           (status, now, error_message, manga_id, chapter_number))
        else:
            cursor.execute("UPDATE queue SET status = ? WHERE manga_id = ? AND chapter_number = ?",
                           (status, manga_id, chapter_number))
        self.conn.commit()

    def clear_tasks_by_status(self, status: str):
        """Удалить задачи с указанным статусом."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM queue WHERE status = ?", (status,))
        self.conn.commit()

    def clear_completed_tasks(self):
        """Удалить все завершённые/отменённые/ошибочные задачи."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM queue WHERE status IN ('completed', 'failed', 'cancelled')")
        self.conn.commit()

    def add_task(self, manga_id: str, chapter_number: str, source_lang: str = "ko") -> bool:
        """Добавить задачу в очередь (alias для add_to_queue)."""
        return self.add_to_queue(manga_id, chapter_number, source_lang)

    def close(self):
        self.conn.close()
