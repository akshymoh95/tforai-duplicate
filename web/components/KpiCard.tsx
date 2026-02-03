import { motion } from "framer-motion";

interface KpiCardProps {
  label: string;
  value: string | number;
  unit?: string;
}

export default function KpiCard({ label, value, unit }: KpiCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      whileHover={{ scale: 1.03, y: -1 }}
      className="relative rounded-lg bg-gradient-to-br from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 px-4 py-2 overflow-hidden group transition-all duration-300 shadow-md hover:shadow-lg border border-blue-500/30"
    >
      {/* Animated shimmer on hover */}
      <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/15 to-white/0 group-hover:from-white/5 group-hover:via-white/20 group-hover:to-white/5 transition-all duration-500 pointer-events-none" />
      
      {/* Animated accent border top */}
      <motion.div
        initial={{ scaleX: 0 }}
        whileHover={{ scaleX: 1 }}
        transition={{ duration: 0.3 }}
        className="absolute top-0 left-0 h-0.5 w-full bg-gradient-to-r from-blue-300 to-cyan-300 origin-left"
      />
      
      <div className="relative z-10">
        <p className="text-xs font-bold tracking-wider text-blue-100 uppercase opacity-85 mb-1">
          {label}
        </p>
        <motion.div
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="font-display text-lg font-bold text-white flex items-baseline gap-1"
        >
          <motion.span
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.4, delay: 0.2, type: "spring", stiffness: 100 }}
          >
            {typeof value === "number" ? value.toFixed(2) : value}
          </motion.span>
          {unit && (
            <span className="text-xs font-semibold text-blue-200">
              {unit}
            </span>
          )}
        </motion.div>
      </div>
    </motion.div>
  );
}
