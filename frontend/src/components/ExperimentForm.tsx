import { useRef, useState } from 'react'
import type { SelectorName, SubmitJob, TrackingMode } from '../types'

interface ExperimentFormProps {
  disabled: boolean
  onSubmit: (input: SubmitJob) => Promise<void>
}

const selectorCopy: Record<SelectorName, string> = {
  uniform: '均匀取样，适合作为对照基线',
  motion: '优先选择运动和场景变化明显的片段',
  query: '根据文字目标选择语义相关片段',
  hybrid: '综合语义、运动和场景多样性',
}

export function ExperimentForm({ disabled, onSubmit }: ExperimentFormProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [prompt, setPrompt] = useState('主要人物')
  const [selector, setSelector] = useState<SelectorName>('hybrid')
  const [trackingMode, setTrackingMode] = useState<TrackingMode>('dense')
  const [frameCount, setFrameCount] = useState(24)
  const [dragging, setDragging] = useState(false)

  const acceptFile = (candidate?: File) => {
    if (candidate) setFile(candidate)
  }

  return (
    <section className="experiment-panel" aria-labelledby="experiment-heading">
      <div className="section-kicker">新建分析</div>
      <h1 id="experiment-heading">说出目标，追踪整段视频。</h1>
      <p className="lede">
        上传视频并描述想找的对象，系统会逐帧追踪，并自动判断目标离场与重新出现。最多支持 4 个目标，每行一个。
      </p>

      <button
        className={`drop-zone ${dragging ? 'dragging' : ''}`}
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          acceptFile(event.dataTransfer.files[0])
        }}
      >
        <span className="drop-index">A</span>
        <span className="drop-copy">
          <strong>{file?.name ?? '拖入视频，或点击选择文件'}</strong>
          <small>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : '支持 MP4、MOV、MKV、WEBM，最大 2 GiB'}</small>
        </span>
        <span className="drop-action">选择文件</span>
      </button>
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        aria-label="选择视频文件"
        accept="video/mp4,video/quicktime,video/webm,.mkv,.avi"
        onChange={(event) => acceptFile(event.target.files?.[0])}
      />

      <label className="field-label" htmlFor="prompt">要追踪什么？</label>
      <textarea
        id="prompt"
        rows={3}
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder={'穿黄色外套的骑行者\n红色自行车'}
      />

      <div className="tracking-mode" role="group" aria-label="输出方式">
        <button
          type="button"
          className={trackingMode === 'dense' ? 'active' : ''}
          onClick={() => setTrackingMode('dense')}
        >
          <span>推荐</span>
          <strong>逐帧追踪</strong>
          <small>离场感知、单区域身份校验与完整掩码</small>
        </button>
        <button
          type="button"
          className={trackingMode === 'analysis' ? 'active' : ''}
          onClick={() => setTrackingMode('analysis')}
        >
          <span>快速</span>
          <strong>关键帧分析</strong>
          <small>仅分割选中的代表帧</small>
        </button>
      </div>

      <details className="advanced-settings">
        <summary>高级设置</summary>
        <div className="form-grid">
          <div>
            <label className="field-label" htmlFor="selector">关键帧选择方式</label>
            <select id="selector" value={selector} onChange={(event) => setSelector(event.target.value as SelectorName)}>
              <option value="hybrid">综合选择（推荐）</option>
              <option value="query">语义相关</option>
              <option value="motion">运动优先</option>
              <option value="uniform">均匀取样（基线）</option>
            </select>
            <p className="field-note">{selectorCopy[selector]}</p>
          </div>
          <div>
            <label className="field-label" htmlFor="frames">分析帧数量 <b>{frameCount} 帧</b></label>
            <input
              id="frames"
              type="range"
              min="8"
              max="48"
              step="4"
              value={frameCount}
              onChange={(event) => setFrameCount(Number(event.target.value))}
            />
            <p className="field-note">默认 24 帧；提高数量会增加分析细度和耗时。</p>
          </div>
        </div>
      </details>

      <button
        className="run-button"
        type="button"
        disabled={disabled || !file || prompt.trim().length < 2}
        onClick={() => file && onSubmit({ file, prompt, selector, trackingMode, frameCount })}
      >
        <span>{disabled ? '正在上传…' : '开始分析'}</span>
        <span>使用 GPU →</span>
      </button>
    </section>
  )
}
