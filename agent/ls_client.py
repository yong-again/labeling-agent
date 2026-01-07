"""
Label Studio API 클라이언트
프로젝트, 태스크, 예측 관리
"""

import logging
from typing import List, Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class LabelStudioClient:
    """Label Studio API 클라이언트"""
    
    def __init__(self, url: str, api_token: str):
        """
        Args:
            url: Label Studio 서버 URL (예: http://localhost:8080)
            api_token: API 토큰
        """
        self.url = url.rstrip("/")
        self.api_token = api_token
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """재시도 전략이 포함된 세션 생성"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """API 요청 헬퍼"""
        url = f"{self.url}/api{endpoint}"
        headers = {"Authorization": f"Token {self.api_token}"}
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {method} {endpoint} - {e}")
            raise
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """프로젝트 목록 조회"""
        logger.info("프로젝트 목록 조회 중...")
        result = self._request("GET", "/projects")
        return result.get("results", [])
    
    def get_project(self, project_id: int) -> Dict[str, Any]:
        """프로젝트 상세 조회"""
        logger.info(f"프로젝트 {project_id} 조회 중...")
        return self._request("GET", f"/projects/{project_id}")
    
    def create_project(
        self,
        title: str,
        label_config: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """프로젝트 생성"""
        logger.info(f"프로젝트 생성 중: {title}")
        data = {
            "title": title,
            "label_config": label_config,
        }
        if description:
            data["description"] = description
        
        return self._request("POST", "/projects", data=data)
    
    def create_tasks(
        self,
        project_id: int,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """태스크 일괄 생성"""
        logger.info(f"프로젝트 {project_id}에 태스크 {len(tasks)}개 생성 중...")
        data = {"project": project_id, "tasks": tasks}
        result = self._request("POST", "/tasks/bulk", data=data)
        return result.get("tasks", [])
    
    def create_task(
        self,
        project_id: int,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """단일 태스크 생성"""
        if not image_path and not image_url:
            raise ValueError("image_path 또는 image_url 중 하나는 필수입니다")
        
        task_data = {}
        if image_url:
            task_data["data"] = {"image": image_url}
        elif image_path:
            # 로컬 파일 경로는 Label Studio가 접근할 수 있어야 함
            # 일반적으로 서버에 마운트된 경로나 URL로 변환 필요
            task_data["data"] = {"image": image_path}
        
        return self.create_tasks(project_id, [task_data])[0]
    
    def upload_predictions(
        self,
        project_id: int,
        predictions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """예측 일괄 업로드"""
        logger.info(f"프로젝트 {project_id}에 예측 {len(predictions)}개 업로드 중...")
        data = {"predictions": predictions}
        result = self._request("POST", f"/projects/{project_id}/predictions", data=data)
        return result.get("predictions", [])
    
    def upload_prediction(
        self,
        project_id: int,
        task_id: int,
        result: List[Dict[str, Any]],
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """단일 예측 업로드"""
        prediction_data = {
            "task": task_id,
            "result": result,
        }
        if score is not None:
            prediction_data["score"] = score
        
        return self.upload_predictions(project_id, [prediction_data])[0]
    
    def get_task(self, task_id: int) -> Dict[str, Any]:
        """태스크 조회"""
        return self._request("GET", f"/tasks/{task_id}")
    
    def get_tasks(self, project_id: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """프로젝트의 태스크 목록 조회"""
        params = {"project": project_id}
        if filters:
            params.update(filters)
        result = self._request("GET", "/tasks", params=params)
        return result.get("results", [])

