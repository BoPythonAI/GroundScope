import type { Health } from '../types'

interface SystemHeaderProps {
  health: Health | null
}

export function SystemHeader({ health }: SystemHeaderProps) {
  const online = health?.cuda_available === true
  return (
    <header className="system-header">
      <div className="wordmark">
        <span className="wordmark-mark" aria-hidden="true">GS</span>
        <div>
          <strong>GROUNDSCOPE</strong>
          <span>语言引导的视频目标分割</span>
        </div>
      </div>
      <div className="system-readouts" aria-label="系统状态">
        <div className="status-pill">
          <i className={online ? 'status-dot online' : 'status-dot'} />
          <span>{online ? 'GPU 在线' : '正在连接'}</span>
        </div>
        <div className="status-pill muted">
          <span>{health?.model_ready ? '模型已加载' : '模型待加载'}</span>
        </div>
      </div>
    </header>
  )
}
