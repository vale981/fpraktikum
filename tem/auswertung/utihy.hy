(import [numpy :as np])
(defn normalize [array]
  (let [tmp (.copy array)]
    (setv tmp (- tmp (.min tmp)))
    (/ tmp (.max tmp))))
