export default function SkeletonCard() {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-xl p-4 flex flex-col gap-3 animate-pulse">
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 bg-gray-700 rounded" />
        <div className="flex-1 space-y-2">
          <div className="h-4 bg-gray-700 rounded w-3/4" />
          <div className="h-3 bg-gray-700 rounded w-1/4" />
        </div>
      </div>
      <div className="flex gap-2">
        <div className="h-5 w-16 bg-gray-700 rounded-full" />
        <div className="h-5 w-14 bg-gray-700 rounded-full" />
      </div>
      <div className="h-3 bg-gray-700 rounded w-2/3" />
    </div>
  )
}
