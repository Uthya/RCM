import { useState, useCallback } from 'react';

interface Props {
  label: string;
  description: string;
  onUpload: (file: File) => Promise<void>;
  accept?: string;
}

export default function FileUpload({ label, description, onUpload, accept = '.edi,.txt,.x12' }: Props) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setFileName(file.name);
    setError(null);
    setUploading(true);
    try {
      await onUpload(file);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [onUpload]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div
      className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        dragActive ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 hover:border-gray-400'
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
      />
      <div className="space-y-2">
        <div className="text-4xl text-gray-400">
          {uploading ? (
            <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600" />
          ) : (
            <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48">
              <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </div>
        <p className="text-lg font-medium text-gray-700">{label}</p>
        <p className="text-sm text-gray-500">{description}</p>
        {fileName && !uploading && (
          <p className="text-sm text-indigo-600 font-medium">Uploaded: {fileName}</p>
        )}
        {error && <p className="text-sm text-red-600">{error}</p>}
      </div>
    </div>
  );
}
