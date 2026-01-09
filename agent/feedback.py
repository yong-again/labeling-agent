"""
HITL 피드백 관리 모듈
Human-In-The-Loop 피드백 저장 및 조회
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import threading

logger = logging.getLogger(__name__)


class FeedbackStatus(str, Enum):
    """피드백 상태"""
    PENDING = "pending"  # 리뷰 대기
    APPROVED = "approved"  # 승인됨 (Pass)
    REJECTED = "rejected"  # 거부됨 (Fail)
    CORRECTED = "corrected"  # 수정됨


@dataclass
class FeedbackItem:
    """피드백 아이템"""
    id: Optional[int]
    image_path: str
    image_id: str
    prompt: str
    status: FeedbackStatus
    labeling_result: Dict[str, Any]  # LabelingResult.to_dict()
    corrections: Optional[Dict[str, Any]]  # 수정된 라벨링 데이터
    created_at: str
    updated_at: str
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = asdict(self)
        result['status'] = self.status.value
        return result


class FeedbackManager:
    """HITL 피드백 관리자"""
    
    def __init__(self, db_path: str = "./feedback.db"):
        """
        Args:
            db_path: SQLite 데이터베이스 경로
        """
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()
        logger.info(f"피드백 DB 초기화: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """스레드별 DB 연결 반환"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_db(self):
        """DB 스키마 초기화"""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                image_id TEXT NOT NULL UNIQUE,
                prompt TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                labeling_result TEXT NOT NULL,
                corrections TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_image_id ON feedback(image_id)
        """)
        conn.commit()
    
    def save_feedback(
        self,
        image_path: str,
        image_id: str,
        prompt: str,
        labeling_result: Dict[str, Any],
        status: FeedbackStatus = FeedbackStatus.PENDING,
        corrections: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        피드백 저장 (upsert)
        
        Args:
            image_path: 이미지 경로
            image_id: 이미지 고유 ID
            prompt: 사용된 프롬프트
            labeling_result: 라벨링 결과 딕셔너리
            status: 피드백 상태
            corrections: 수정된 라벨링 데이터
            notes: 추가 메모
            
        Returns:
            피드백 ID
        """
        conn = self._get_connection()
        now = datetime.now().isoformat()
        
        try:
            conn.execute("""
                INSERT INTO feedback (image_path, image_id, prompt, status, 
                                      labeling_result, corrections, notes,
                                      created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    status = excluded.status,
                    labeling_result = excluded.labeling_result,
                    corrections = excluded.corrections,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
            """, (
                image_path,
                image_id,
                prompt,
                status.value,
                json.dumps(labeling_result),
                json.dumps(corrections) if corrections else None,
                notes,
                now,
                now,
            ))
            conn.commit()
            
            # 생성된 ID 조회
            cursor = conn.execute(
                "SELECT id FROM feedback WHERE image_id = ?",
                (image_id,)
            )
            row = cursor.fetchone()
            feedback_id = row['id'] if row else -1
            
            logger.info(f"피드백 저장: id={feedback_id}, image_id={image_id}, status={status.value}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"피드백 저장 실패: {e}")
            raise
    
    def update_status(
        self,
        image_id: str,
        status: FeedbackStatus,
        corrections: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        피드백 상태 업데이트
        
        Args:
            image_id: 이미지 ID
            status: 새 상태
            corrections: 수정된 라벨링 데이터
            notes: 추가 메모
            
        Returns:
            성공 여부
        """
        conn = self._get_connection()
        now = datetime.now().isoformat()
        
        try:
            if corrections:
                conn.execute("""
                    UPDATE feedback 
                    SET status = ?, corrections = ?, notes = ?, updated_at = ?
                    WHERE image_id = ?
                """, (status.value, json.dumps(corrections), notes, now, image_id))
            else:
                conn.execute("""
                    UPDATE feedback 
                    SET status = ?, notes = ?, updated_at = ?
                    WHERE image_id = ?
                """, (status.value, notes, now, image_id))
            
            conn.commit()
            updated = conn.total_changes > 0
            
            if updated:
                logger.info(f"피드백 상태 업데이트: image_id={image_id}, status={status.value}")
            else:
                logger.warning(f"피드백을 찾을 수 없음: image_id={image_id}")
            
            return updated
            
        except Exception as e:
            logger.error(f"피드백 상태 업데이트 실패: {e}")
            raise
    
    def get_feedback(self, image_id: str) -> Optional[FeedbackItem]:
        """이미지 ID로 피드백 조회"""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM feedback WHERE image_id = ?",
            (image_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_feedback(row)
    
    def get_pending_reviews(self, limit: int = 100) -> List[FeedbackItem]:
        """리뷰 대기 중인 피드백 목록"""
        conn = self._get_connection()
        cursor = conn.execute(
            """SELECT * FROM feedback 
               WHERE status = ? 
               ORDER BY created_at ASC 
               LIMIT ?""",
            (FeedbackStatus.PENDING.value, limit)
        )
        
        return [self._row_to_feedback(row) for row in cursor.fetchall()]
    
    def get_by_status(
        self, 
        status: FeedbackStatus, 
        limit: int = 100
    ) -> List[FeedbackItem]:
        """상태별 피드백 목록"""
        conn = self._get_connection()
        cursor = conn.execute(
            """SELECT * FROM feedback 
               WHERE status = ? 
               ORDER BY updated_at DESC 
               LIMIT ?""",
            (status.value, limit)
        )
        
        return [self._row_to_feedback(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """피드백 통계"""
        conn = self._get_connection()
        
        # 상태별 카운트
        cursor = conn.execute("""
            SELECT status, COUNT(*) as count 
            FROM feedback 
            GROUP BY status
        """)
        
        stats = {
            "total": 0,
            "by_status": {},
        }
        
        for row in cursor.fetchall():
            stats["by_status"][row['status']] = row['count']
            stats["total"] += row['count']
        
        # 오늘 생성된 피드백 수
        today = datetime.now().strftime("%Y-%m-%d")
        cursor = conn.execute(
            "SELECT COUNT(*) as count FROM feedback WHERE created_at LIKE ?",
            (f"{today}%",)
        )
        stats["today"] = cursor.fetchone()['count']
        
        return stats
    
    def export_training_data(
        self,
        status: Optional[FeedbackStatus] = None,
        include_corrections: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Continuous Learning을 위한 학습 데이터 내보내기
        
        Args:
            status: 특정 상태만 내보내기 (None이면 approved + corrected)
            include_corrections: 수정된 데이터 포함 여부
            
        Returns:
            학습 데이터 리스트
        """
        conn = self._get_connection()
        
        if status:
            cursor = conn.execute(
                "SELECT * FROM feedback WHERE status = ?",
                (status.value,)
            )
        else:
            # 승인 또는 수정된 데이터만
            cursor = conn.execute(
                "SELECT * FROM feedback WHERE status IN (?, ?)",
                (FeedbackStatus.APPROVED.value, FeedbackStatus.CORRECTED.value)
            )
        
        training_data = []
        for row in cursor.fetchall():
            item = {
                "image_path": row['image_path'],
                "prompt": row['prompt'],
            }
            
            # 수정된 데이터가 있으면 수정본 사용
            if include_corrections and row['corrections']:
                item["labels"] = json.loads(row['corrections'])
            else:
                item["labels"] = json.loads(row['labeling_result'])
            
            training_data.append(item)
        
        logger.info(f"학습 데이터 내보내기: {len(training_data)}개 샘플")
        return training_data
    
    def _row_to_feedback(self, row: sqlite3.Row) -> FeedbackItem:
        """DB row를 FeedbackItem으로 변환"""
        return FeedbackItem(
            id=row['id'],
            image_path=row['image_path'],
            image_id=row['image_id'],
            prompt=row['prompt'],
            status=FeedbackStatus(row['status']),
            labeling_result=json.loads(row['labeling_result']),
            corrections=json.loads(row['corrections']) if row['corrections'] else None,
            notes=row['notes'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )
    
    def close(self):
        """DB 연결 종료"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
