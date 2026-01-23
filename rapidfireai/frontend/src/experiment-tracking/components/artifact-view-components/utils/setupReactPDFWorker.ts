export function setupReactPDFWorker(pdfjsInstance: { GlobalWorkerOptions: { workerSrc: string } }) {
  pdfjsInstance.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
  ).toString();
}
